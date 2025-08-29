"""
Tests for model deployment and rollback functionality.
Tests the versioning, deployment, and rollback capabilities of the AutomatedRetrainer.
"""

import os
import json
import uuid
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Mock the dependencies to avoid import errors
with patch.dict('sys.modules', {
    'joblib': MagicMock(),
    'sklearn': MagicMock(),
    'mlflow': MagicMock(),
    'core.nlu.model_loader': MagicMock(),
    'core.logging.query_logger': MagicMock(),
    'core.filtering.hybrid_filter': MagicMock()
}):
    from scripts.retrain_classifier import AutomatedRetrainer

class TestDeployRollback:
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def retrainer(self, temp_dir):
        """Create an AutomatedRetrainer instance with temp directory."""
        with patch('scripts.retrain_classifier.MODEL_DIR', temp_dir):
            return AutomatedRetrainer()
    
    def test_version_save_and_deploy(self, retrainer, temp_dir):
        """Test saving a model version and deploying it."""
        # Mock classifier and vectorizer
        mock_classifier = MagicMock()
        mock_vectorizer = MagicMock()
        mock_labels = ["label1", "label2", "label3"]
        
        # Set up retrainer training metrics
        retrainer.training_metrics = {
            "accuracy": 0.85,
            "f1_macro": 0.82,
            "f1_weighted": 0.84
        }
        
        # Mock joblib.dump
        with patch('joblib.dump') as mock_dump:
            # Save model without deploy
            version_id = retrainer._save_model(
                mock_classifier, 
                mock_vectorizer, 
                mock_labels, 
                deploy=False
            )
            
            # Check version_id format
            assert isinstance(version_id, str)
            assert len(version_id) > 10  # Should have timestamp + uuid
            
            # Check version directory exists
            versions_dir = os.path.join(temp_dir, "versions")
            version_path = os.path.join(versions_dir, version_id)
            assert os.path.exists(version_path)
            
            # Check files were created
            assert os.path.exists(os.path.join(version_path, "label_mappings.json"))
            assert os.path.exists(os.path.join(version_path, "metrics.json"))
            
            # Check versions index
            versions_index = os.path.join(versions_dir, "versions.json")
            assert os.path.exists(versions_index)
            
            with open(versions_index) as f:
                index = json.load(f)
                assert version_id in index
                assert index[version_id]["deployed"] == False
                assert index[version_id]["path"] == version_path
    
    def test_deploy_version(self, retrainer, temp_dir):
        """Test deploying a specific model version."""
        # Create a mock version first
        versions_dir = os.path.join(temp_dir, "versions")
        os.makedirs(versions_dir, exist_ok=True)
        
        version_id = "20231201_120000_abcd1234"
        version_path = os.path.join(versions_dir, version_id)
        os.makedirs(version_path, exist_ok=True)
        
        # Create version index
        index = {
            version_id: {
                "created": datetime.now().isoformat(),
                "path": version_path,
                "metrics": {"accuracy": 0.85},
                "deployed": False,
                "deploy_mode": None
            }
        }
        versions_index = os.path.join(versions_dir, "versions.json")
        with open(versions_index, 'w') as f:
            json.dump(index, f)
        
        # Mock core.nlu.model_loader.reload_current_model
        with patch('core.nlu.model_loader.reload_current_model') as mock_reload:
            # Deploy the version
            success = retrainer.deploy_version(version_id, mode="full")
            
            assert success == True
            
            # Check symlink was created
            current_symlink = os.path.join(temp_dir, "current")
            assert os.path.islink(current_symlink)
            assert os.readlink(current_symlink) == version_path
            
            # Check index was updated
            with open(versions_index) as f:
                updated_index = json.load(f)
                assert updated_index[version_id]["deployed"] == True
                assert updated_index[version_id]["deploy_mode"] == "full"
    
    def test_rollback_to_version(self, retrainer, temp_dir):
        """Test rolling back to a previous model version."""
        # Create two mock versions
        versions_dir = os.path.join(temp_dir, "versions")
        os.makedirs(versions_dir, exist_ok=True)
        
        old_version = "20231201_100000_old12345"
        new_version = "20231201_120000_new12345"
        
        old_path = os.path.join(versions_dir, old_version)
        new_path = os.path.join(versions_dir, new_version)
        os.makedirs(old_path, exist_ok=True)
        os.makedirs(new_path, exist_ok=True)
        
        # Create version index with new version deployed
        index = {
            old_version: {
                "created": "2023-12-01T10:00:00",
                "path": old_path,
                "metrics": {"accuracy": 0.80},
                "deployed": False,
                "deploy_mode": None
            },
            new_version: {
                "created": "2023-12-01T12:00:00", 
                "path": new_path,
                "metrics": {"accuracy": 0.85},
                "deployed": True,
                "deploy_mode": "full"
            }
        }
        versions_index = os.path.join(versions_dir, "versions.json")
        with open(versions_index, 'w') as f:
            json.dump(index, f)
        
        # Create current symlink pointing to new version
        current_symlink = os.path.join(temp_dir, "current")
        os.symlink(new_path, current_symlink)
        
        # Mock core.nlu.model_loader.reload_current_model
        with patch('core.nlu.model_loader.reload_current_model') as mock_reload:
            # Rollback to old version
            success = retrainer.rollback_to_version(old_version)
            
            assert success == True
            
            # Check symlink now points to old version
            assert os.readlink(current_symlink) == old_path
            
            # Check index was updated
            with open(versions_index) as f:
                updated_index = json.load(f)
                assert updated_index[old_version]["deployed"] == True
                assert updated_index[old_version]["deploy_mode"] == "rollback"
                assert updated_index[new_version]["deployed"] == False
    
    def test_list_versions(self, retrainer, temp_dir):
        """Test listing all model versions."""
        # Create mock versions
        versions_dir = os.path.join(temp_dir, "versions")
        os.makedirs(versions_dir, exist_ok=True)
        
        index = {
            "version1": {
                "created": "2023-12-01T10:00:00",
                "metrics": {"accuracy": 0.80},
                "deployed": False
            },
            "version2": {
                "created": "2023-12-01T12:00:00",
                "metrics": {"accuracy": 0.85},
                "deployed": True
            }
        }
        versions_index = os.path.join(versions_dir, "versions.json")
        with open(versions_index, 'w') as f:
            json.dump(index, f)
        
        # List versions
        versions = retrainer.list_versions()
        
        assert isinstance(versions, dict)
        assert "version1" in versions
        assert "version2" in versions
        assert versions["version2"]["deployed"] == True
    
    def test_get_version_info(self, retrainer, temp_dir):
        """Test getting information about a specific version."""
        # Create mock version
        versions_dir = os.path.join(temp_dir, "versions")
        os.makedirs(versions_dir, exist_ok=True)
        
        version_id = "test_version_123"
        index = {
            version_id: {
                "created": "2023-12-01T10:00:00",
                "metrics": {"accuracy": 0.80},
                "deployed": False
            }
        }
        versions_index = os.path.join(versions_dir, "versions.json")
        with open(versions_index, 'w') as f:
            json.dump(index, f)
        
        # Get version info
        info = retrainer.get_version_info(version_id)
        
        assert info is not None
        assert info["created"] == "2023-12-01T10:00:00"
        assert info["metrics"]["accuracy"] == 0.80
        
        # Test non-existent version
        none_info = retrainer.get_version_info("non_existent")
        assert none_info is None
    
    def test_abtest_management(self, retrainer, temp_dir):
        """Test A/B testing start/stop functionality."""
        version_id = "test_version_abtest"
        fraction = 0.05
        
        # Start A/B test
        success = retrainer.start_abtest(version_id, fraction)
        assert success == True
        
        # Check abtest.json was created
        abtest_file = os.path.join(temp_dir, "abtest.json")
        assert os.path.exists(abtest_file)
        
        # Check content
        with open(abtest_file) as f:
            config = json.load(f)
            assert config["version"] == version_id
            assert config["fraction"] == fraction
        
        # Get A/B test status
        status = retrainer.get_abtest_status()
        assert status is not None
        assert status["version"] == version_id
        assert status["fraction"] == fraction
        
        # Check if A/B test is active
        assert retrainer.is_abtest_active() == True
        
        # Stop A/B test
        stop_success = retrainer.stop_abtest()
        assert stop_success == True
        assert not os.path.exists(abtest_file)
        assert retrainer.is_abtest_active() == False
        
        # Get status after stop
        status_after = retrainer.get_abtest_status()
        assert status_after is None
    
    def test_deploy_nonexistent_version(self, retrainer, temp_dir):
        """Test deploying a version that doesn't exist."""
        success = retrainer.deploy_version("nonexistent_version")
        assert success == False
    
    def test_rollback_nonexistent_version(self, retrainer, temp_dir):
        """Test rolling back to a version that doesn't exist."""
        success = retrainer.rollback_to_version("nonexistent_version")
        assert success == False
    
    def test_atomic_deployment(self, retrainer, temp_dir):
        """Test that deployment is atomic (uses temporary symlink)."""
        # Create a mock version
        versions_dir = os.path.join(temp_dir, "versions")
        os.makedirs(versions_dir, exist_ok=True)
        
        version_id = "atomic_test_version"
        version_path = os.path.join(versions_dir, version_id)
        os.makedirs(version_path, exist_ok=True)
        
        # Create version index
        index = {
            version_id: {
                "created": datetime.now().isoformat(),
                "path": version_path,
                "metrics": {"accuracy": 0.85},
                "deployed": False,
                "deploy_mode": None
            }
        }
        versions_index = os.path.join(versions_dir, "versions.json")
        with open(versions_index, 'w') as f:
            json.dump(index, f)
        
        # Mock core.nlu.model_loader.reload_current_model to track calls
        with patch('core.nlu.model_loader.reload_current_model') as mock_reload:
            # Deploy the version
            success = retrainer.deploy_version(version_id)
            
            assert success == True
            # Verify hot reload was called
            mock_reload.assert_called_once()
            
            # Check no temporary files are left behind
            temp_files = [f for f in os.listdir(temp_dir) if f.startswith("current_tmp_")]
            assert len(temp_files) == 0
