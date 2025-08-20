#!/usr/bin/env python3
"""
Cron Job Setup Script for Automated Retraining
-----------------------------------------------
This script helps set up automated monthly retraining of the ML classifier.
It creates cron jobs and provides monitoring capabilities.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CronSetup")

class CronRetrainingSetup:
    """Setup and manage automated retraining cron jobs"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.absolute()
        self.cron_log = self.project_root / "data/logs/cron_retrain.log"
        self.cron_status_file = self.project_root / "data/logs/cron_status.json"
        
    def setup_monthly_retraining(self, hour: int = 3, minute: int = 0, day: int = 1):
        """Set up monthly retraining cron job"""
        
        logger.info("🚀 Setting up Monthly Retraining Cron Job")
        logger.info("=" * 50)
        
        try:
            # Create cron job entry
            cron_entry = self._create_cron_entry(hour, minute, day)
            
            # Check if cron job already exists
            if self._cron_job_exists():
                logger.warning("⚠️  Cron job already exists!")
                self._show_existing_cron()
                return False
            
            # Add cron job
            if self._add_cron_job(cron_entry):
                logger.info("✅ Monthly retraining cron job added successfully!")
                logger.info(f"   Schedule: {day}st of every month at {hour:02d}:{minute:02d}")
                logger.info(f"   Command: {cron_entry}")
                
                # Save cron status
                self._save_cron_status(cron_entry, hour, minute, day)
                
                # Create monitoring script
                self._create_monitoring_script()
                
                return True
            else:
                logger.error("❌ Failed to add cron job")
                return False
                
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    def _create_cron_entry(self, hour: int, minute: int, day: int) -> str:
        """Create cron job entry string"""
        
        retrain_script = self.project_root / "scripts" / "retrain_classifier.py"
        log_file = self.project_root / "data/logs/cron_retrain.log"
        
        # Set environment variables
        env_vars = f"PYTHONPATH={self.project_root}"
        
        # Create command
        command = f"{minute} {hour} {day} * * {env_vars} python {retrain_script} >> {log_file} 2>&1"
        
        return command
    
    def _cron_job_exists(self) -> bool:
        """Check if retraining cron job already exists"""
        
        try:
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            if result.returncode == 0:
                cron_content = result.stdout
                return "retrain_classifier.py" in cron_content
            return False
        except Exception as e:
            logger.error(f"Failed to check existing cron jobs: {e}")
            return False
    
    def _show_existing_cron(self):
        """Show existing cron jobs"""
        
        try:
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("📋 Existing cron jobs:")
                for line in result.stdout.split('\n'):
                    if line.strip() and "retrain_classifier.py" in line:
                        logger.info(f"   {line}")
            else:
                logger.info("No existing cron jobs found")
        except Exception as e:
            logger.error(f"Failed to show existing cron jobs: {e}")
    
    def _add_cron_job(self, cron_entry: str) -> bool:
        """Add cron job to user's crontab"""
        
        try:
            # Get current crontab
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            current_crontab = result.stdout if result.returncode == 0 else ""
            
            # Add new entry with comment
            new_entry = f"# Monthly AI Model Retraining - Added {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{cron_entry}\n"
            updated_crontab = current_crontab + new_entry
            
            # Write updated crontab
            with open('/tmp/temp_crontab', 'w') as f:
                f.write(updated_crontab)
            
            # Install new crontab
            result = subprocess.run(['crontab', '/tmp/temp_crontab'], capture_output=True, text=True)
            
            # Clean up temp file
            os.remove('/tmp/temp_crontab')
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to add cron job: {e}")
            return False
    
    def _save_cron_status(self, cron_entry: str, hour: int, minute: int, day: int):
        """Save cron job status for monitoring"""
        
        try:
            status = {
                'cron_entry': cron_entry,
                'schedule': {
                    'hour': hour,
                    'minute': minute,
                    'day': day,
                    'description': f"{day}st of every month at {hour:02d}:{minute:02d}"
                },
                'setup_date': datetime.now().isoformat(),
                'status': 'active',
                'last_run': None,
                'next_run': self._calculate_next_run(hour, minute, day)
            }
            
            os.makedirs(self.cron_status_file.parent, exist_ok=True)
            with open(self.cron_status_file, 'w') as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save cron status: {e}")
    
    def _calculate_next_run(self, hour: int, minute: int, day: int) -> str:
        """Calculate next run time for cron job"""
        
        now = datetime.now()
        
        # Find next occurrence
        if now.day >= day:
            # Next month
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=day, hour=hour, minute=minute, second=0, microsecond=0)
            else:
                next_run = now.replace(month=now.month + 1, day=day, hour=hour, minute=minute, second=0, microsecond=0)
        else:
            # This month
            next_run = now.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
        
        return next_run.isoformat()
    
    def _create_monitoring_script(self):
        """Create monitoring script for cron job status"""
        
        try:
            monitor_script = self.project_root / "scripts" / "monitor_cron.py"
            
            script_content = f'''#!/usr/bin/env python3
"""
Cron Job Monitoring Script
---------------------------
Monitor the status of automated retraining cron jobs.
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

def check_cron_status():
    """Check cron job status and health"""
    
    project_root = Path(__file__).parent.parent
    status_file = project_root / "data/logs/cron_status.json"
    log_file = project_root / "data/logs/cron_retrain.log"
    
    print("📊 Cron Job Monitoring Report")
    print("=" * 40)
    
    # Check cron job status
    if status_file.exists():
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        print(f"Status: {status.get('status', 'Unknown')}")
        print(f"Schedule: {status.get('schedule', {}).get('description', 'Unknown')}")
        print(f"Setup Date: {status.get('setup_date', 'Unknown')}")
        print(f"Next Run: {status.get('next_run', 'Unknown')}")
    else:
        print("❌ No cron status file found")
        return
    
    # Check if cron job is active
    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode == 0 and "retrain_classifier.py" in result.stdout:
            print("✅ Cron job is active")
        else:
            print("❌ Cron job not found in crontab")
    except Exception as e:
        print(f"❌ Failed to check crontab: {e}")
    
    # Check log file
    if log_file.exists():
        log_size = log_file.stat().st_size
        log_time = datetime.fromtimestamp(log_file.stat().st_mtime)
        
        print(f"\\n📝 Log File Status:")
        print(f"   Size: {log_size} bytes")
        print(f"   Last Modified: {log_time}")
        
        # Show last few lines
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"\\n📋 Last Log Entries:")
                    for line in lines[-5:]:
                        print(f"   {line.strip()}")
        except Exception as e:
            print(f"   Failed to read log: {e}")
    else:
        print("\\n❌ Log file not found")

if __name__ == "__main__":
    check_cron_status()
'''
            
            with open(monitor_script, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(monitor_script, 0o755)
            
            logger.info(f"✅ Monitoring script created: {monitor_script}")
            
        except Exception as e:
            logger.error(f"Failed to create monitoring script: {e}")
    
    def remove_cron_job(self):
        """Remove retraining cron job"""
        
        logger.info("🗑️  Removing Monthly Retraining Cron Job")
        logger.info("=" * 50)
        
        try:
            # Get current crontab
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.info("No existing crontab found")
                return True
            
            current_crontab = result.stdout
            
            # Remove retraining entries
            lines = current_crontab.split('\n')
            filtered_lines = []
            
            skip_next = False
            for line in lines:
                if "retrain_classifier.py" in line:
                    skip_next = True
                    continue
                elif skip_next and line.startswith('#'):
                    skip_next = False
                    continue
                elif not skip_next:
                    filtered_lines.append(line)
            
            # Write updated crontab
            if filtered_lines:
                with open('/tmp/temp_crontab', 'w') as f:
                    f.write('\\n'.join(filtered_lines) + '\\n')
                
                result = subprocess.run(['crontab', '/tmp/temp_crontab'], capture_output=True, text=True)
                os.remove('/tmp/temp_crontab')
                
                if result.returncode == 0:
                    logger.info("✅ Cron job removed successfully")
                    
                    # Update status file
                    if self.cron_status_file.exists():
                        with open(self.cron_status_file, 'r') as f:
                            status = json.load(f)
                        status['status'] = 'removed'
                        status['removed_date'] = datetime.now().isoformat()
                        
                        with open(self.cron_status_file, 'w') as f:
                            json.dump(status, f, indent=2)
                    
                    return True
                else:
                    logger.error("❌ Failed to remove cron job")
                    return False
            else:
                # Remove entire crontab
                subprocess.run(['crontab', '-r'])
                logger.info("✅ All cron jobs removed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to remove cron job: {e}")
            return False
    
    def show_status(self):
        """Show current cron job status"""
        
        logger.info("📊 Cron Job Status")
        logger.info("=" * 30)
        
        # Check if cron job exists
        if self._cron_job_exists():
            logger.info("✅ Retraining cron job is active")
        else:
            logger.info("❌ No retraining cron job found")
        
        # Show status file
        if self.cron_status_file.exists():
            with open(self.cron_status_file, 'r') as f:
                status = json.load(f)
            
            print(f"\\n📋 Cron Job Details:")
            print(f"   Status: {status.get('status', 'Unknown')}")
            print(f"   Schedule: {status.get('schedule', {}).get('description', 'Unknown')}")
            print(f"   Setup Date: {status.get('setup_date', 'Unknown')}")
            print(f"   Next Run: {status.get('next_run', 'Unknown')}")
        else:
            print("\\n❌ No cron status file found")
        
        # Show next steps
        print(f"\\n💡 Next Steps:")
        print(f"   1. Monitor cron job: python {self.project_root}/scripts/monitor_cron.py")
        print(f"   2. Check retraining status: python {self.project_root}/scripts/retrain_classifier.py --status")
        print(f"   3. View logs: tail -f {self.project_root}/data/logs/cron_retrain.log")

def main():
    """Main function for command-line usage"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Automated Monthly Retraining')
    parser.add_argument('--setup', action='store_true', help='Setup monthly retraining cron job')
    parser.add_argument('--remove', action='store_true', help='Remove retraining cron job')
    parser.add_argument('--status', action='store_true', help='Show cron job status')
    parser.add_argument('--hour', type=int, default=3, help='Hour for retraining (0-23, default: 3)')
    parser.add_argument('--minute', type=int, default=0, help='Minute for retraining (0-59, default: 0)')
    parser.add_argument('--day', type=int, default=1, help='Day of month for retraining (1-31, default: 1)')
    
    args = parser.parse_args()
    
    setup = CronRetrainingSetup()
    
    if args.setup:
        success = setup.setup_monthly_retraining(args.hour, args.minute, args.day)
        if success:
            print("\\n🎉 Monthly retraining setup complete!")
            print("\\n📚 What happens next:")
            print("   • Every month on the 1st at 3:00 AM, the system will:")
            print("     1. Check for new training data from user queries")
            print("     2. Automatically retrain the ML classifier")
            print("     3. Update the model with new knowledge")
            print("     4. Log all activities for monitoring")
            print("\\n   • Monitor the process:")
            print("     python scripts/monitor_cron.py")
            print("     python scripts/retrain_classifier.py --status")
        else:
            print("\\n❌ Setup failed. Check logs for details.")
            sys.exit(1)
    
    elif args.remove:
        success = setup.remove_cron_job()
        if success:
            print("\\n✅ Cron job removed successfully")
        else:
            print("\\n❌ Failed to remove cron job")
            sys.exit(1)
    
    elif args.status:
        setup.show_status()
    
    else:
        parser.print_help()
        print("\\n💡 Example usage:")
        print("   python scripts/setup_cron_retraining.py --setup")
        print("   python scripts/setup_cron_retraining.py --status")
        print("   python scripts/setup_cron_retraining.py --remove")

if __name__ == "__main__":
    main()
