import json
import logging
import asyncio
import aiohttp
from typing import List, Dict, Optional
import yaml
import os

logger = logging.getLogger(__name__)

class LLMParser:
    def __init__(self, config_path: str = "config/llm.yaml"):
        """Initialize LLM parser with configuration."""
        self.config = self._load_config(config_path)
        self.session = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load LLM configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded LLM config: {config['provider']} - {config['model']}")
            return config
        except Exception as e:
            logger.error(f"Failed to load LLM config: {e}")
            # Fallback config
            return {
                "provider": "openrouter",
                "model": "deepseek/deepseek-r1-0528:free",
                "api_key": "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",
                "api_url": "https://openrouter.ai/api/v1/chat/completions",
                "max_tokens": 256,
                "temperature": 0.0,
                "timeout": 30,
                "retry_attempts": 3,
                "site_url": "http://localhost:8000",
                "site_name": "AI Mood Food Recommender"
            }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    def _create_prompt(self, query: str, taxonomy: List[str]) -> str:
        """Create the prompt for LLM classification."""
        system_prompt = """You are a food intent classifier. Given a query and labels, return JSON only.
Example: {"labels": ["goal_comfort", "sensory_warming"]}"""

        # Use the most common/relevant labels for better matching
        core_labels = [
            "goal_comfort", "sensory_warming", "health_illness", "goal_hydration",
            "sensory_refreshing", "flavor_sweet", "flavor_spicy", "flavor_salty",
            "emotional_comfort", "goal_energy", "meal_dinner", "meal_lunch",
            "occasion_home", "texture_creamy", "cuisine_asian", "goal_quick"
        ]
        
        # Use core labels if available, otherwise first 16 from taxonomy
        if any(label in taxonomy for label in core_labels):
            available_labels = [label for label in core_labels if label in taxonomy][:16]
        else:
            available_labels = taxonomy[:16] if len(taxonomy) > 16 else taxonomy
        
        user_prompt = f"""Query: "{query}"
Choose from: {', '.join(available_labels)}
Response:"""

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
    
    async def _call_openrouter_api(self, prompt: Dict) -> Optional[Dict]:
        """Call OpenRouter API with the given prompt."""
        session = await self._get_session()
        
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        # Add optional OpenRouter headers for rankings
        if self.config.get('site_url'):
            headers["HTTP-Referer"] = self.config['site_url']
        if self.config.get('site_name'):
            headers["X-Title"] = self.config['site_name']
        
        payload = {
            "model": self.config['model'],
            "messages": prompt["messages"],
            "max_tokens": self.config.get('max_tokens', 256),
            "temperature": self.config.get('temperature', 0.0)
        }
        
        api_url = self.config.get('api_url', 'https://openrouter.ai/api/v1/chat/completions')
        
        for attempt in range(self.config.get('retry_attempts', 3)):
            try:
                async with session.post(
                    api_url,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"OpenRouter API success: {data}")
                        return data
                    else:
                        error_text = await response.text()
                        logger.warning(f"OpenRouter API error (attempt {attempt + 1}): {response.status} - {error_text}")
                        logger.warning(f"Request payload: {payload}")
                        logger.warning(f"Request headers: {headers}")
                        
            except Exception as e:
                logger.warning(f"OpenRouter API call failed (attempt {attempt + 1}): {e}")
                
            if attempt < self.config.get('retry_attempts', 3) - 1:
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return None
    
    def _parse_llm_response(self, response: Dict) -> List[str]:
        """Parse LLM response and extract labels."""
        try:
            if not response or 'choices' not in response:
                logger.warning(f"Invalid LLM response structure: {response}")
                return []
            
            message = response['choices'][0]['message']
            content = message.get('content', '').strip()
            reasoning = message.get('reasoning', '').strip()
            
            logger.info(f"LLM content: {content}")
            logger.info(f"LLM reasoning: {reasoning[:200]}..." if reasoning else "No reasoning")
            
            # For DeepSeek R1, try reasoning field first, then content
            text_to_parse = content if content else reasoning
            
            if not text_to_parse:
                logger.warning("No content or reasoning in LLM response")
                return []
            
            # Try to extract JSON from the response
            text_to_parse = text_to_parse.strip()
            if text_to_parse.startswith('```json'):
                text_to_parse = text_to_parse[7:]
            if text_to_parse.endswith('```'):
                text_to_parse = text_to_parse[:-3]
            text_to_parse = text_to_parse.strip()
            
            # Look for JSON patterns in the text
            json_start = text_to_parse.find('{"labels":')
            if json_start == -1:
                json_start = text_to_parse.find('{ "labels":')
            
            if json_start != -1:
                # Find the end of the JSON object
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(text_to_parse[json_start:], json_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                json_text = text_to_parse[json_start:json_end]
                logger.info(f"Extracted JSON: {json_text}")
                
                try:
                    parsed = json.loads(json_text)
                    labels = parsed.get('labels', [])
                    
                    if not isinstance(labels, list):
                        logger.warning("Labels is not a list in LLM response")
                        return []
                    
                    logger.info(f"LLM parsed labels: {labels}")
                    return labels
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted JSON: {e}")
            
            # If no JSON found, try to parse the entire text
            try:
                parsed = json.loads(text_to_parse)
                labels = parsed.get('labels', [])
                
                if not isinstance(labels, list):
                    logger.warning("Labels is not a list in LLM response")
                    return []
                
                logger.info(f"LLM parsed labels: {labels}")
                return labels
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                logger.error(f"Response text: {text_to_parse[:500]}...")
                return []
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []
    
    async def classify_with_llm(self, query: str, taxonomy: List[str]) -> List[str]:
        """Classify user query using LLM semantic parsing."""
        try:
            logger.info(f"Classifying query with LLM: '{query}'")
            
            # Create prompt
            prompt = self._create_prompt(query, taxonomy)
            
            # Call LLM API
            response = await self._call_openrouter_api(prompt)
            
            if response is None:
                logger.warning("LLM API call failed, returning empty labels")
                return []
            
            # Parse response
            labels = self._parse_llm_response(response)
            
            logger.info(f"LLM classification result: {labels}")
            return labels
            
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return []
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

# Global instance for easy access
_llm_parser = None

async def get_llm_parser() -> LLMParser:
    """Get or create global LLM parser instance."""
    global _llm_parser
    if _llm_parser is None:
        _llm_parser = LLMParser()
    return _llm_parser

async def classify_with_llm(query: str, taxonomy: List[str]) -> List[str]:
    """Convenience function to classify query with LLM."""
    parser = await get_llm_parser()
    return await parser.classify_with_llm(query, taxonomy)
