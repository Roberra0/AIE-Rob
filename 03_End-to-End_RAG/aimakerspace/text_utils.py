import os
import warnings
import logging
from typing import List

# Suppress ALL warnings before importing pdfplumber
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import pdfplumber


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_pdf()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt/.pdf file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())
    
    def load_pdf(self, pdf_path=None):
        """Extract text from PDF file using pdfplumber with smart column detection"""
        if pdf_path is None:
            pdf_path = self.path
            
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    # Try smart column detection first
                    page_text = self._extract_text_smart(page)
                    if page_text:
                        text += page_text + "\n"
                        
                self.documents.append(text)
        except Exception as e:
            raise ValueError(f"Error reading PDF file {pdf_path}: {str(e)}")
    
    def _extract_text_smart(self, page):
        """Smart text extraction that detects column layout"""
        # First, try normal extraction
        normal_text = page.extract_text()
        if not normal_text:
            return ""
        
        # Check if text looks like it has column mixing (has scattered contact info)
        lines = normal_text.split('\n')
        mixed_lines = 0
        total_lines = len(lines)
        
        # Look for patterns that indicate column mixing
        for line in lines:
            # Check for phone numbers, emails, or short isolated text (typical of right columns)
            if (len(line.strip()) < 20 and 
                (any(char.isdigit() for char in line) or 
                 '@' in line or 
                 line.strip() in ['Apply Now', 'Contact Us', 'MAP', 'BED / BATH'])):
                mixed_lines += 1
        
        # If more than 20% of lines look like column mixing, use column extraction
        if mixed_lines > total_lines * 0.2:
            return self._extract_left_column_only(page)
        else:
            # Use normal extraction for single-column or well-formatted PDFs
            return normal_text
    
    def _extract_left_column_only(self, page):
        """Extract text from left column only for multi-column layouts"""
        width = page.width
        height = page.height
        
        # Try different column ratios and pick the one with most coherent text
        best_text = ""
        best_ratio = 0.6
        
        for ratio in [0.5, 0.6, 0.7, 0.8]:
            left_column = page.crop((0, 0, width * ratio, height))
            left_text = left_column.extract_text()
            
            if left_text:
                # Simple heuristic: prefer text with longer average line length
                lines = left_text.split('\n')
                avg_line_length = sum(len(line.strip()) for line in lines) / len(lines)
                
                if avg_line_length > len(best_text.split('\n')[0]) if best_text else 0:
                    best_text = left_text
                    best_ratio = ratio
        
        return best_text
    
    def load_pdf_advanced(self, pdf_path=None):
        """Alternative PDF extraction method using text layout analysis"""
        if pdf_path is None:
            pdf_path = self.path
            
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    # Extract words with their positions
                    words = page.extract_words()
                    
                    # Filter words that are in the left portion of the page
                    left_words = [word for word in words if word['x0'] < page.width * 0.6]
                    
                    # Sort by y-coordinate (top to bottom) then x-coordinate (left to right)
                    left_words.sort(key=lambda w: (w['top'], w['x0']))
                    
                    # Reconstruct text
                    page_text = ""
                    current_line = ""
                    current_y = None
                    
                    for word in left_words:
                        if current_y is None or abs(word['top'] - current_y) < 5:  # Same line
                            current_line += word['text'] + " "
                            current_y = word['top']
                        else:  # New line
                            page_text += current_line.strip() + "\n"
                            current_line = word['text'] + " "
                            current_y = word['top']
                    
                    # Add the last line
                    if current_line:
                        page_text += current_line.strip() + "\n"
                    
                    text += page_text + "\n"
                        
                self.documents.append(text)
        except Exception as e:
            raise ValueError(f"Error reading PDF file {pdf_path}: {str(e)}")

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())
                elif file.endswith(".pdf"):
                    self.load_pdf(os.path.join(root, file))

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
