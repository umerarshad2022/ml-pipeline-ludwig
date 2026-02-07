"""
Ludwig Pipeline Engine
YAML-based ML pipeline orchestration with Ludwig AI
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
# Setup logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
@dataclass
class PipelineConfig:
   """Pipeline configuration"""
   name: str
   description: str
   metadata: Dict[str, Any]
   data: Dict[str, Any]
   ludwig: Dict[str, Any]
   steps: list
   experiment: Optional[Dict[str, Any]] = None
   artifacts: Optional[Dict[str, Any]] = None
   deployment: Optional[Dict[str, Any]] = None
class LudwigPipelineEngine:
   """
   Main pipeline engine for Ludwig-based ML workflows.
  
   Features:
   - YAML-based configuration
   - Ludwig AI integration
   - Experiment tracking
   - Artifact management
   """
  
   def __init__(self, config_path: str):
       """
       Initialize pipeline engine.
      
       Args:
           config_path: Path to YAML configuration file
       """
       self.config_path = Path(config_path)
       self.config = None
       self.context = {}
       self.results = {}
      
       logger.info(f"Initializing Ludwig Pipeline Engine")
       logger.info(f"Config: {self.config_path}")
      
   def load_config(self) -> PipelineConfig:
       """Load and parse YAML configuration"""
       logger.info(f"Loading configuration from {self.config_path}")
      
       with open(self.config_path, 'r') as f:
           config_dict = yaml.safe_load(f)
      
       # Parse into dataclass
       self.config = PipelineConfig(
           name=config_dict.get('name', 'Unnamed Pipeline'),
           description=config_dict.get('description', ''),
           metadata=config_dict.get('metadata', {}),
           data=config_dict.get('data', {}),
           ludwig=config_dict.get('ludwig', {}),
           steps=config_dict.get('steps', []),
           experiment=config_dict.get('experiment'),
           artifacts=config_dict.get('artifacts'),
           deployment=config_dict.get('deployment')
       )
      
       logger.info(f"✅ Loaded pipeline: {self.config.name}")
       logger.info(f"   Steps: {len(self.config.steps)}")
      
       return self.config
  
   def validate_config(self):
       """Validate pipeline configuration"""
       logger.info("Validating configuration...")
      
       # Required fields
       if not self.config.ludwig:
           raise ValueError("Ludwig configuration is required")
      
       if not self.config.ludwig.get('input_features'):
           raise ValueError("Ludwig input_features are required")
      
       if not self.config.ludwig.get('output_features'):
           raise ValueError("Ludwig output_features are required")
      
       logger.info("✅ Configuration valid")
  
   def execute(self) -> Dict[str, Any]:
       """
       Execute the pipeline.
      
       Returns:
           dict: Pipeline execution results
       """
       # Load config FIRST before accessing it
       if not self.config:
           self.load_config()
      
       logger.info("=" * 80)
       logger.info(f"  EXECUTING PIPELINE: {self.config.name}")
       logger.info("=" * 80)
      
       # Validate
       self.validate_config()
      
       # Execute each step
       for i, step in enumerate(self.config.steps, 1):
           step_name = step.get('name', f'step-{i}')
           step_type = step.get('type')
          
           logger.info(f"\n[{i}/{len(self.config.steps)}] {step_name}")
           logger.info(f"     Type: {step_type}")
           logger.info("-" * 80)
          
           try:
               result = self._execute_step(step)
               self.results[step_name] = {
                   'status': 'success',
                   'result': result
               }
               logger.info(f"✅ {step_name} completed")
              
           except Exception as e:
               logger.error(f"❌ {step_name} failed: {e}")
               self.results[step_name] = {
                   'status': 'failed',
                   'error': str(e)
               }
               raise
      
       logger.info("\n" + "=" * 80)
       logger.info("  ✅ PIPELINE EXECUTION COMPLETE")
       logger.info("=" * 80)
      
       return self.results
  
   def _execute_step(self, step: Dict[str, Any]) -> Any:
       """Execute a single pipeline step"""
       step_type = step.get('type')
       step_config = step.get('config', {})
      
       # Import handlers dynamically
       if step_type == 'data_loader':
           from pipelines.handlers.data_loader import DataLoaderHandler
           handler = DataLoaderHandler(step_config, self.context)
          
       elif step_type == 'unsupervised_labeling':
           from pipelines.handlers.data_loader import UnsupervisedLabelingHandler
           handler = UnsupervisedLabelingHandler(step_config, self.context)
          
       elif step_type == 'ludwig_train':
           from pipelines.handlers.ludwig_handler import LudwigTrainHandler
           handler = LudwigTrainHandler(step_config, self.context, self.config.ludwig)
          
       elif step_type == 'ludwig_evaluate':
           from pipelines.handlers.ludwig_handler import LudwigEvaluateHandler
           handler = LudwigEvaluateHandler(step_config, self.context)
          
       elif step_type == 'ludwig_visualize':
           from pipelines.handlers.ludwig_visualize import LudwigVisualizeHandler
           handler = LudwigVisualizeHandler(step_config, self.context)
          
       else:
           raise ValueError(f"Unknown step type: {step_type}")
      
       # Execute handler
       result = handler.execute()
      
       # Update context
       if result:
           self.context[step.get('name')] = result
      
       return result
def main():
   """CLI entry point"""
   import sys
  
   if len(sys.argv) < 2:
       print("Usage: python -m pipelines.engine <config.yaml>")
       sys.exit(1)
  
   config_path = sys.argv[1]
  
   # Run pipeline
   engine = LudwigPipelineEngine(config_path)
   results = engine.execute()
  
   print(f"\n✅ Pipeline completed")
   print(f"   Results: {len(results)} steps")
if __name__ == '__main__':
   main()
