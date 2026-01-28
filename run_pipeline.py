"""
Ludwig Pipeline Runner
Simple CLI to run Ludwig-based ML pipelines
"""
import sys
import argparse
from pathlib import Path
from pipelines.engine import LudwigPipelineEngine


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Run Ludwig-based ML pipeline from YAML configuration'
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='Path to pipeline YAML configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='artifacts',
        help='Output directory for artifacts (default: artifacts/)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for training (if available)'
    )
    
    args = parser.parse_args()
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("  LUDWIG ML PIPELINE")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output: {args.output}")
    print(f"GPU:    {'Enabled' if args.gpu else 'Disabled'}")
    print("=" * 80)
    
    # Run pipeline
    try:
        engine = LudwigPipelineEngine(str(config_path))
        results = engine.execute()
        
        print("\n" + "=" * 80)
        print("  ✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSteps completed: {len(results)}")
        
        for step_name, step_result in results.items():
            status = step_result.get('status', 'unknown')
            print(f"  • {step_name}: {status}")
        
        print(f"\n📁 Artifacts saved to: {args.output}/")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
