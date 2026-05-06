from dataclasses import dataclass
from typing import Optional
import subprocess
import os


@dataclass
class DeploymentResult:
    """Deployment result for an architecture."""
    url: Optional[str] = None
    platform: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


class ModelDeployment:
    """Deploys architectures to various platforms."""

    PLATFORMS = {
        "huggingface": {
            "name": "HuggingFace Hub",
            "requires": "HF_TOKEN",
            "description": "Upload to HF model hub",
        },
        "replicate": {
            "name": "Replicate",
            "requires": "REPLICATE_API_TOKEN",
            "description": "Deploy to Replicate for API serving",
        },
        "vllm": {
            "name": "vLLM OpenAI API",
            "requires": "Docker",
            "description": "Local vLLM server",
        },
    }

    @classmethod
    def deploy_to_huggingface(
        cls,
        gene,
        repo_id: str,
        token: Optional[str] = None,
    ) -> DeploymentResult:
        if not token:
            token = os.environ.get("HF_TOKEN")

        if not token:
            return DeploymentResult(
                error="HF_TOKEN not set. Get from https://huggingface.co/settings/tokens"
            )

        try:
            subprocess.run(
                ["huggingface-cli", "whoami"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return DeploymentResult(
                error="HuggingFace CLI not installed. Run: pip install huggingface-hub"
            )

        return DeploymentResult(
            url=f"https://huggingface.co/{repo_id}",
            platform="huggingface",
            status="upload_ready",
        )

    @classmethod
    def deploy_to_vllm(
        cls,
        gene,
        model_path: str = "./exports/model",
    ) -> DeploymentResult:
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return DeploymentResult(
                error="Docker not installed. Install from https://docker.com"
            )

        return DeploymentResult(
            url="http://localhost:8000/v1/chat/completions",
            platform="vllm",
            status="startable",
        )

    @classmethod
    def deploy_to_replicate(
        cls,
        gene,
        model_id: Optional[str] = None,
    ) -> DeploymentResult:
        token = os.environ.get("REPLICATE_API_TOKEN")

        if not token:
            return DeploymentResult(
                error="REPLICATE_API_TOKEN not set. Get from https://replicate.com/settings"
            )

        return DeploymentResult(
            url=f"https://api.replicate.com/v1/models/{model_id}" if model_id else None,
            platform="replicate",
            status="api_ready",
        )

    @classmethod
    def get_deployment_instructions(
        cls,
        platform: str,
    ) -> str:
        instructions = {
            "huggingface": """# Deploy to HuggingFace

1. Install HF CLI: pip install huggingface-hub
2. Login: huggingface-cli login
3. Upload model: python -c "from src.archgene.exporter import Exporter; Exporter.to_huggingface(gene, 'your-repo-id')"

Or use the CLI:
  python main.py export --format huggingface --repo-id your-username/model-name
""",
            "vllm": """# Deploy with vLLM

1. Install vLLM: pip install vllm
2. Start server:
   vllm serve meta-llama/Llama-2-7b-hf --tensor-parallel-size 1
3. API available at: http://localhost:8000/v1/chat/completions
""",
            "replicate": """# Deploy to Replicate

1. Install Replicate: pip install replicate
2. Create model on Replicate.com
3. Push model: python main.py export --format replicate
""",
        }
        return instructions.get(platform, "Unknown platform")


if __name__ == "__main__":
    print("Supported platforms:")
    for key, info in ModelDeployment.PLATFORMS.items():
        print(f"  {key}: {info['description']}")

    print("\nDeployment instructions:")
    print(ModelDeployment.get_deployment_instructions("huggingface"))