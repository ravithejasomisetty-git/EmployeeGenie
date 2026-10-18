"""Shared settings and LangChain integrations for OpenAI and Pinecone.

The Pinecone SDK handles index administration and ingestion verification.
Importing this module does not create clients or make service requests.
"""
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


@dataclass
class AIModels:
    """Configured LangChain chat and embedding clients."""

    chat: ChatOpenAI
    embeddings: OpenAIEmbeddings


class PreparedBatchEmbeddings(Embeddings):
    """Embed contextual inputs while storing readable passages.

    The adapter preserves title and heading context in vectors without adding
    those prefixes to stored passage text.
    """

    def __init__(self, embeddings, chunks):
        """Keep one ordered batch and the configured LangChain embedding wrapper."""
        self.embeddings, self.chunks = embeddings, chunks

    def embed_documents(self, texts):
        """Verify batch order and embed each contextual input.

        The complete prepared batch must be supplied in its original order.
        """
        if texts != [chunk['metadata']['text'] for chunk in self.chunks]:
            raise ValueError('Prepared embedding batch does not match uploaded passages')
        return self.embeddings.embed_documents([chunk['embedding_text'] for chunk in self.chunks])

    def embed_query(self, text):
        """Delegate a query to the underlying embedding model."""
        return self.embeddings.embed_query(text)


def vector_store(index, embeddings, namespace):
    """Wrap an existing index without creating it.

    text_key='text' makes existing ingested records readable without re-ingestion.
    Search results are LangChain Documents whose page_content contains the stored
    text; their metadata contains the other fields and id identifies the vector.
    """
    return PineconeVectorStore(index=index, embedding=embeddings, text_key='text', namespace=namespace)


@dataclass(frozen=True)
class Settings:
    """Immutable ingestion and chat configuration suitable for resource cache keys.

    Credentials stay in environment variables, outside this object and reports.
    """
    index: str
    namespace: str
    embedding_model: str
    chat_model: str
    report: str
    cloud: str
    region: str
    # One shared value controls document vectors, query vectors, and index creation.
    embedding_dimensions: int = 512

    def __post_init__(self):
        """Validate configured embedding dimensions."""
        if self.embedding_dimensions <= 0:
            raise ValueError('EMBEDDING_DIMENSIONS must be a positive integer')

    @classmethod
    def read(cls):
        """Load .env without overriding process environment values.

        The chat readiness check rejects an empty namespace.
        """
        load_dotenv()
        return cls(os.getenv('PINECONE_INDEX', 'employee-policy-helper'),
                   os.getenv('PINECONE_NAMESPACE', ''),
                   os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
                   os.getenv('CHAT_MODEL', 'gpt-4.1-mini'),
                   os.getenv('INGESTION_REPORT', 'artifacts/ingestion-report.json'),
                   os.getenv('PINECONE_CLOUD', 'aws'), os.getenv('PINECONE_REGION', 'us-east-1'),
                   int(os.getenv('EMBEDDING_DIMENSIONS', '512')))


def clients(settings, create=False):
    """Return (LangChain AIModels, Pinecone index connection).

    create is permission to create a missing index, not an instruction to replace
    one. Only ingestion passes create=True. Chat must never create a missing index.
    A Pinecone index holds vectors; a namespace partitions records within it.
    Network/authentication errors are allowed to propagate to the calling layer.
    """
    missing = [key for key in ('OPENAI_API_KEY', 'PINECONE_API_KEY') if not os.getenv(key)]
    if missing:
        raise ValueError('Set these values in .env: ' + ', '.join(missing))
    # LangChain configures the underlying OpenAI SDK timeout and eligible retries.
    ai = AIModels(
        chat=ChatOpenAI(model=settings.chat_model, timeout=45, max_retries=2),
        embeddings=OpenAIEmbeddings(model=settings.embedding_model,
                                   dimensions=settings.embedding_dimensions,
                                   request_timeout=45, max_retries=2),
    )
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    index_exists = pc.has_index(settings.index)

    # Chat needs an existing index; only ingestion is allowed to create one.
    if not index_exists and not create:
        raise ValueError('Pinecone index does not exist. Run ingestion first.')

    if not index_exists:
        # The index must expect exactly the configured number of embedding values.
        # Cosine similarity compares vector direction to rank semantic resemblance.
        pc.create_index(
            name=settings.index,
            dimension=settings.embedding_dimensions,
            metric='cosine',
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
        )
    # Index creation is asynchronous; poll until ready.
    for _ in range(30):
        description = pc.describe_index(settings.index)
        if description.dimension != settings.embedding_dimensions or description.metric != 'cosine':
            raise ValueError(
                f'Index {settings.index!r} uses {description.dimension} dimensions and '
                f'{description.metric}; expected {settings.embedding_dimensions} and cosine. '
                'Set PINECONE_INDEX to a new index name and re-ingest. A new namespace '
                'alone cannot change index dimensions. Existing data has not been deleted.'
            )
        if description.status['ready']:
            return ai, pc.Index(host=description.host)
        time.sleep(2)
    raise TimeoutError('Pinecone index did not become ready')


def embed(ai, texts, model, dimensions=512):
    """Convert text into vectors with the requested number of floating-point values.

    Embeddings represent meaning for similarity search; they are not generated
    answers. Both stored passages and incoming questions must use the same model.
    LangChain returns vectors in input order so they stay paired with their text.
    """
    # The LangChain model is configured once in clients(); prevent accidental drift.
    if ai.embeddings.model != model or ai.embeddings.dimensions != dimensions:
        raise ValueError('Embedding configuration does not match the requested model/dimensions')
    return ai.embeddings.embed_documents(texts)


def retry(operation):
    """Retry HTTP 429, HTTP 5xx and supported connection or timeout failures.

    Run the operation at most four times with exponential backoff. Other errors
    propagate immediately; not every SDK exception type is retryable.
    """
    for attempt in range(4):
        try:
            return operation()
        except Exception as exc:
            # SDKs use different attribute names. getattr supplies a default when absent.
            status = getattr(exc, 'status', None) or getattr(exc, 'status_code', None)
            if attempt == 3 or not (status == 429 or isinstance(exc, (TimeoutError, ConnectionError))
                                    or isinstance(status, int) and status >= 500):
                raise
            # Exponential backoff waits 1, then 2, then 4 seconds between attempts.
            time.sleep(2 ** attempt)
