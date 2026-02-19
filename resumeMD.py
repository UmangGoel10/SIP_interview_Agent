import os
import logging
from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError

# Set up logging to track what happens during execution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_resume_to_markdown(file_path):
    """
    Converts a PDF or Image resume to Markdown with error handling.
    """
    try:
        # 1. Validation: Does the file exist?
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} was not found.")

        # 2. Initialize Converter
        # Note: You can move this outside the function if processing multiple files
        # to avoid reloading models every time.
        converter = DocumentConverter()

        logger.info(f"Starting conversion for: {file_path}")

        # 3. Perform Conversion
        result = converter.convert(file_path)

        # 4. Export and Return
        markdown_output = result.document.export_to_markdown()
        
        logger.info("Conversion successful.")
        return markdown_output

    except FileNotFoundError as fnf_error:
        logger.error(f"File Error: {fnf_error}")
        return None

    except ConversionError as conv_error:
        # This catches Docling-specific issues (e.g., corrupt PDF, unsupported layout)
        logger.error(f"Docling failed to convert the document: {conv_error}")
        return None

    except Exception as e:
        # Catch-all for unexpected system errors (Memory, Permissions, etc.)
        logger.error(f"An unexpected error occurred: {e.with_traceback()}")
        return None








