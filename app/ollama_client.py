
import requests


class OllamaClient:
    def __init__(
        self,
        host="http://127.0.0.1:11434",
        embed_model="nomic-embed-text",
        gen_model="gemma2:2b",
    ):
        self.host = host.rstrip("/")
        self.embed_model = embed_model
        self.gen_model = gen_model

    def is_available(self):
        try:
            response = requests.get(
                f"{self.host}/api/tags", timeout=2
            )
            return response.ok
        except requests.RequestException:
            return False

    def embed(self, text):
        try:
            # Current Ollama embedding endpoint.
            response = requests.post(
                f"{self.host}/api/embed",
                json={
                    "model": self.embed_model,
                    "input": text,
                },
                timeout=60,
            )

            if response.status_code == 404:
                # Compatibility fallback for older Ollama versions.
                response = requests.post(
                    f"{self.host}/api/embeddings",
                    json={
                        "model": self.embed_model,
                        "prompt": text,
                    },
                    timeout=60,
                )

            response.raise_for_status()
            data = response.json()

            if "embeddings" in data:
                vectors = data["embeddings"]
                if not vectors:
                    return []
                return vectors[0]

            return data.get("embedding", [])

        except (requests.RequestException, ValueError):
            return []

    def generate(self, prompt):
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.gen_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=180,
            )
            response.raise_for_status()
            return response.json().get("response", "")

        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"
