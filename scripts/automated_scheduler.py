#!/usr/bin/env python3
"""
Automated Retraining Scheduler
------------------------------
This module provides automated scheduling for model retraining using APScheduler.
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import json

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    print("APScheduler not available. Install with: pip install apscheduler")

logger = logging.getLogger(__name__)

class AutomatedRetrainingScheduler:
    """Automated scheduler for model retraining"""
    
    def __init__(self, api_client=None):
        self.scheduler = None
        self.api_client = api_client
        self.is_running = False
        self.schedule_config = {
            'weekly_retrain': {
                'enabled': True,
                'day_of_week': 'sun',  # Sunday
                'hour': 3,  # 3 AM
                'minute': 0
            },
            'monthly_retrain': {
                'enabled': True,
                'day': 1,  # 1st of month
                'hour': 2,  # 2 AM
                'minute': 0
            },
            'adaptive_retrain': {
                'enabled': True,
                'min_samples': 100,  # Retrain when 100+ new samples
                'min_days': 7,  # Or at least 7 days
                'check_interval_hours': 6  # Check every 6 hours
            }
        }
    
    def start_scheduler(self):
        """Start the automated scheduler"""
        if not APSCHEDULER_AVAILABLE:
            logger.error("APScheduler not available. Cannot start automated scheduler.")
            return False
        
        try:
            self.scheduler = AsyncIOScheduler()
            
            # Add weekly retraining job
            if self.schedule_config['weekly_retrain']['enabled']:
                self.scheduler.add_job(
                    self._weekly_retrain_job,
                    CronTrigger(
                        day_of_week=self.schedule_config['weekly_retrain']['day_of_week'],
                        hour=self.schedule_config['weekly_retrain']['hour'],
                        minute=self.schedule_config['weekly_retrain']['minute']
                    ),
                    id='weekly_retrain',
                    name='Weekly Model Retraining',
                    replace_existing=True
                )
                logger.info("✅ Weekly retraining scheduled")
            
            # Add monthly retraining job
            if self.schedule_config['monthly_retrain']['enabled']:
                self.scheduler.add_job(
                    self._monthly_retrain_job,
                    CronTrigger(
                        day=self.schedule_config['monthly_retrain']['day'],
                        hour=self.schedule_config['monthly_retrain']['hour'],
                        minute=self.schedule_config['monthly_retrain']['minute']
                    ),
                    id='monthly_retrain',
                    name='Monthly Model Retraining',
                    replace_existing=True
                )
                logger.info("✅ Monthly retraining scheduled")
            
            # Add adaptive retraining job
            if self.schedule_config['adaptive_retrain']['enabled']:
                self.scheduler.add_job(
                    self._adaptive_retrain_job,
                    IntervalTrigger(
                        hours=self.schedule_config['adaptive_retrain']['check_interval_hours']
                    ),
                    id='adaptive_retrain',
                    name='Adaptive Model Retraining',
                    replace_existing=True
                )
                logger.info("✅ Adaptive retraining scheduled")
            
            # Start the scheduler
            self.scheduler.start()
            self.is_running = True
            
            logger.info("🚀 Automated retraining scheduler started successfully")
            logger.info(f"   Next weekly retrain: {self._get_next_run_time('weekly_retrain')}")
            logger.info(f"   Next monthly retrain: {self._get_next_run_time('monthly_retrain')}")
            logger.info(f"   Next adaptive check: {self._get_next_run_time('adaptive_retrain')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False
    
    def stop_scheduler(self):
        """Stop the automated scheduler"""
        if self.scheduler and self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("🛑 Automated retraining scheduler stopped")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        if not self.scheduler:
            return {
                'status': 'not_started',
                'jobs': [],
                'next_runs': {}
            }
        
        jobs = []
        next_runs = {}
        
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
            next_runs[job.id] = job.next_run_time.isoformat() if job.next_run_time else None
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'jobs': jobs,
            'next_runs': next_runs,
            'config': self.schedule_config
        }
    
    def _get_next_run_time(self, job_id: str) -> Optional[str]:
        """Get next run time for a specific job"""
        if not self.scheduler:
            return None
        
        job = self.scheduler.get_job(job_id)
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
        return None
    
    async def _weekly_retrain_job(self):
        """Weekly retraining job"""
        logger.info("🔄 Weekly retraining job triggered")
        await self._trigger_retraining("cron_weekly")
    
    async def _monthly_retrain_job(self):
        """Monthly retraining job"""
        logger.info("🔄 Monthly retraining job triggered")
        await self._trigger_retraining("cron_monthly")
    
    async def _adaptive_retrain_job(self):
        """Adaptive retraining job - checks conditions and retrains if needed"""
        logger.info("🔍 Adaptive retraining check triggered")
        
        try:
            # Check if retraining is needed
            should_retrain, reason = await self._check_retraining_conditions()
            
            if should_retrain:
                logger.info(f"🔄 Adaptive retraining triggered: {reason}")
                await self._trigger_retraining("cron_adaptive", reason=reason)
            else:
                logger.info(f"⏸️ Adaptive retraining skipped: {reason}")
                
        except Exception as e:
            logger.error(f"Adaptive retraining check failed: {e}")
    
    async def _check_retraining_conditions(self) -> tuple[bool, str]:
        """Check if retraining conditions are met"""
        try:
            # Check dataset size
            dataset_size = await self._get_dataset_size()
            
            # Check last retrain time
            last_retrain = await self._get_last_retrain_time()
            
            # Check new samples count
            new_samples = await self._get_new_samples_count()
            
            config = self.schedule_config['adaptive_retrain']
            
            # Condition 1: Enough new samples
            if new_samples >= config['min_samples']:
                return True, f"Sufficient new samples ({new_samples} >= {config['min_samples']})"
            
            # Condition 2: Enough time has passed
            if last_retrain:
                days_since_retrain = (datetime.now() - last_retrain).days
                if days_since_retrain >= config['min_days']:
                    return True, f"Enough time passed ({days_since_retrain} days >= {config['min_days']})"
            
            return False, f"Insufficient conditions (samples: {new_samples}, days: {days_since_retrain if last_retrain else 'unknown'})"
            
        except Exception as e:
            logger.error(f"Error checking retraining conditions: {e}")
            return False, f"Error checking conditions: {e}"
    
    async def _trigger_retraining(self, trigger_type: str, reason: str = ""):
        """Trigger retraining via API"""
        try:
            if self.api_client:
                # Use API client if available
                response = await self.api_client.post("/retrain")
                logger.info(f"Retraining triggered via API client: {response.status_code}")
            else:
                # Fallback to direct script execution
                await self._run_retraining_script(trigger_type, reason)
                
        except Exception as e:
            logger.error(f"Failed to trigger retraining: {e}")
    
    async def _run_retraining_script(self, trigger_type: str, reason: str = ""):
        """Run retraining script directly"""
        try:
            import subprocess
            import asyncio
            
            # Run the retraining script
            cmd = ["python", "scripts/retrain_classifier.py", "--force"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✅ Retraining completed successfully (trigger: {trigger_type})")
                if reason:
                    logger.info(f"   Reason: {reason}")
            else:
                logger.error(f"❌ Retraining failed (trigger: {trigger_type})")
                logger.error(f"   Error: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Failed to run retraining script: {e}")
    
    async def _get_dataset_size(self) -> int:
        """Get current dataset size"""
        try:
            # Check training dataset file
            dataset_file = Path("data/logs/training_dataset.jsonl")
            if dataset_file.exists():
                with open(dataset_file, 'r') as f:
                    return sum(1 for line in f)
            return 0
        except Exception as e:
            logger.error(f"Error getting dataset size: {e}")
            return 0
    
    async def _get_last_retrain_time(self) -> Optional[datetime]:
        """Get last retraining time"""
        try:
            from scripts.retrain_classifier import AutomatedRetrainer
            retrainer = AutomatedRetrainer()
            status = retrainer.get_retrain_status()
            
            last_retrain_str = status.get('last_retrain')
            if last_retrain_str and last_retrain_str != 'Never':
                return datetime.fromisoformat(last_retrain_str.replace('Z', '+00:00'))
            
            return None
        except Exception as e:
            logger.error(f"Error getting last retrain time: {e}")
            return None
    
    async def _get_new_samples_count(self) -> int:
        """Get count of new samples since last retrain"""
        try:
            last_retrain = await self._get_last_retrain_time()
            if not last_retrain:
                return 0
            
            # Count samples added after last retrain
            new_samples = 0
            dataset_file = Path("data/logs/training_dataset.jsonl")
            
            if dataset_file.exists():
                with open(dataset_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            timestamp_str = entry.get('timestamp', '')
                            if timestamp_str:
                                entry_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                if entry_time > last_retrain:
                                    new_samples += 1
                        except:
                            continue
            
            return new_samples
        except Exception as e:
            logger.error(f"Error getting new samples count: {e}")
            return 0
    
    def update_schedule_config(self, config: Dict[str, Any]):
        """Update scheduler configuration"""
        self.schedule_config.update(config)
        logger.info("📝 Scheduler configuration updated")
        
        # Restart scheduler if running
        if self.is_running:
            self.stop_scheduler()
            self.start_scheduler()

# Global scheduler instance
scheduler_instance = None

def get_scheduler() -> AutomatedRetrainingScheduler:
    """Get global scheduler instance"""
    global scheduler_instance
    if scheduler_instance is None:
        scheduler_instance = AutomatedRetrainingScheduler()
    return scheduler_instance

def start_automated_scheduler(api_client=None) -> bool:
    """Start the automated scheduler"""
    scheduler = get_scheduler()
    scheduler.api_client = api_client
    return scheduler.start_scheduler()

def stop_automated_scheduler():
    """Stop the automated scheduler"""
    scheduler = get_scheduler()
    scheduler.stop_scheduler()

def get_scheduler_status() -> Dict[str, Any]:
    """Get scheduler status"""
    scheduler = get_scheduler()
    return scheduler.get_scheduler_status()

if __name__ == "__main__":
    # Test the scheduler
    import asyncio
    
    async def test_scheduler():
        scheduler = AutomatedRetrainingScheduler()
        
        # Test status
        status = scheduler.get_scheduler_status()
        print("Scheduler status:", status)
        
        # Test starting
        success = scheduler.start_scheduler()
        print(f"Scheduler started: {success}")
        
        if success:
            # Wait a bit
            await asyncio.sleep(2)
            
            # Check status again
            status = scheduler.get_scheduler_status()
            print("Running scheduler status:", status)
            
            # Stop
            scheduler.stop_scheduler()
            print("Scheduler stopped")
    
    asyncio.run(test_scheduler())
