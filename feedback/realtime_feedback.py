import asyncio
from asyncio import Queue
import json
from typing import AsyncGenerator

class RealTimeFeedbackProcessor:
    def __init__(self):
        self.feedback_queue = Queue()
        self.processing_tasks = []
        self.batch_size = 10
        self.batch_timeout = 5.0  # seconds
        
    async def start_processing(self):
        """Start the real-time feedback processing system"""
        # Start multiple processing workers
        for i in range(3):  # 3 worker tasks
            task = asyncio.create_task(self._feedback_worker(f"worker-{i}"))
            self.processing_tasks.append(task)
        
        # Start batch processor
        batch_task = asyncio.create_task(self._batch_processor())
        self.processing_tasks.append(batch_task)
    
    async def submit_feedback(self, feedback_data: Dict):
        """Submit feedback for processing"""
        await self.feedback_queue.put(feedback_data)
    
    async def _feedback_worker(self, worker_id: str):
        """Worker task to process individual feedback items"""
        feedback_collector = FeedbackCollector()
        
        while True:
            try:
                # Get feedback from queue
                feedback_data = await self.feedback_queue.get()
                
                # Process feedback
                processed_feedback = await feedback_collector.collect_feedback(feedback_data)
                
                # Send to learning system
                await self._send_to_learning_system(processed_feedback)
                
                # Update user profile
                await self._update_user_profile(processed_feedback)
                
                # Mark task as done
                self.feedback_queue.task_done()
                
            except Exception as e:
                print(f"Error in feedback worker {worker_id}: {e}")
                await asyncio.sleep(1)
    
    async def _batch_processor(self):
        """Process feedback in batches for efficiency"""
        batch = []
        last_process_time = asyncio.get_event_loop().time()
        
        while True:
            try:
                # Try to get feedback with timeout
                try:
                    feedback = await asyncio.wait_for(
                        self.feedback_queue.get(), 
                        timeout=1.0
                    )
                    batch.append(feedback)
                except asyncio.TimeoutError:
                    pass
                
                current_time = asyncio.get_event_loop().time()
                
                # Process batch if it's full or timeout reached
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_process_time > self.batch_timeout)):
                    
                    await self._process_feedback_batch(batch)
                    batch = []
                    last_process_time = current_time
                
            except Exception as e:
                print(f"Error in batch processor: {e}")
                await asyncio.sleep(1)
    
    async def _process_feedback_batch(self, feedback_batch: List[Dict]):
        """Process a batch of feedback for analytics and learning"""
        if not feedback_batch:
            return
        
        # Group feedback by user for efficient processing
        user_feedback = {}
        for feedback in feedback_batch:
            user_id = feedback.get("user_id")
            if user_id:
                if user_id not in user_feedback:
                    user_feedback[user_id] = []
                user_feedback[user_id].append(feedback)
        
        # Process each user's feedback
        for user_id, user_feedback_list in user_feedback.items():
            await self._process_user_feedback_batch(user_id, user_feedback_list)
    
    async def _send_to_learning_system(self, processed_feedback: Dict):
        """Send processed feedback to learning system"""
        learning_service = LearningService()
        await learning_service.process_feedback(processed_feedback)
    
    async def _update_user_profile(self, processed_feedback: Dict):
        """Update user profile based on feedback"""
        profile_service = UserProfileService()
        await profile_service.update_from_feedback(processed_feedback)