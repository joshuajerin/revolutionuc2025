# Configure logging to use stdout instead of stderr
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Write to stdout instead of stderr
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
// ... existing code ... 