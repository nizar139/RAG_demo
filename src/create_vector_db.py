import argparse
import glob
import torch
from pathlib import Path

from typing import List

from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from src.config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_markdown_file(file_path: str) -> str:
    """
    Loads a markdown file and returns its content.
    Args:
        file_path (str): The path to the markdown file.
    Returns:
        str: The content of the markdown file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            markdown_content = file.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1", errors="replace") as file:
            markdown_content = file.read()
    return markdown_content


def markdown_to_chunks(
    file_path: str,
    header_levels: list = None,
    chunk_size: int = 256,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Splits a markdown file into chunks based on specified header levels.

    Args:
        file_path (str): The path to the markdown file.
        header_levels (list, optional): List of header levels to use for splitting. Defaults to ['#', '##', '###'].
    Returns:
        vector_db: The created vector database.
    """
    if header_levels is None:
        header_levels = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    # Initialize the MarkdownHeaderTextSplitter with specified header levels
    markdown_splitter = MarkdownHeaderTextSplitter(header_levels)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Read the markdown file
    markdown_content = load_markdown_file(file_path)
    file_name = Path(file_path).stem

    # Split the content into chunks based on headers
    chunks = markdown_splitter.split_text(markdown_content)
    chunks = text_splitter.split_documents(chunks)

    for chunk in chunks:
        metadata = {
            "document": file_name,
            "source": file_path,
        }
        chunk.metadata.update(metadata)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Create a vector database from markdown files."
    )
    parser.add_argument(
        "--docs_path", type=str, required=True, help="Path to the markdown documents."
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the vector database."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=CHUNK_SIZE, help="Size of each text chunk."
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Overlap size between text chunks.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=EMBED_MODEL,
        help="Embedding model to use.",
    )

    args = parser.parse_args()

    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name="my_collection",
        embedding_function=embedding_model,
        persist_directory=args.save_path,
    )
    vector_store.reset_collection()

    for docs in glob.iglob(f"{args.docs_path}/*.md"):
        chunks = markdown_to_chunks(
            docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        _ = vector_store.add_documents(documents=chunks)

    results = vector_store.similarity_search_with_score(
        "If the QRC fails during boot, what is the expected system behavior and recommended recovery steps?"
    )

    for i in range(3):
        doc, score = results[i]
        print(f"Result {i + 1} - Score: {score}\n")
        print(doc)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
