import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import xml.etree.ElementTree as ET
from datetime import datetime
import json

# Text processing imports
try:
    import ebooklib
    from ebooklib import epub
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False
    print("⚠️ ebooklib not available. Install with: pip install ebooklib")

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PyPDF2 not available. Install with: pip install PyPDF2")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("⚠️ BeautifulSoup4 not available. Install with: pip install beautifulsoup4")

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    print("⚠️ chardet not available. Install with: pip install chardet")

# MOBI support is basic - no external library needed for current implementation
MOBI_AVAILABLE = True

# Audio processing
import numpy as np
from scipy.io.wavfile import write
import soundfile as sf

class EBookConverter:
    """Convert eBooks to audiobooks using TTS engines."""
    
    SUPPORTED_FORMATS = {
        '.epub': 'EPUB eBook',
        '.pdf': 'PDF Document', 
        '.txt': 'Plain Text',
        '.html': 'HTML Document (Best for automatic chapter detection)',
        '.htm': 'HTML Document (Best for automatic chapter detection)'
    }
    
    def __init__(self, output_dir: str = "audiobooks"):
        """Initialize the eBook converter.
        
        Args:
            output_dir: Directory to save generated audiobooks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Chapter detection patterns
        self.chapter_patterns = [
            r'^Chapter\s+\d+',
            r'^CHAPTER\s+\d+',
            r'^Ch\.\s*\d+',
            r'^\d+\.\s+',
            r'^Part\s+\d+',
            r'^PART\s+\d+',
            r'^Section\s+\d+',
            r'^Book\s+\d+',
            r'^Volume\s+\d+',
            r'^\*\*\*',
            r'^---+',
            r'^===+'
        ]
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        if not CHARDET_AVAILABLE:
            return 'utf-8'
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8') or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def extract_epub_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract content from EPUB file with chapter detection.
        
        Returns:
            Tuple of (full_text, chapters_list)
        """
        if not EPUB_AVAILABLE or not BS4_AVAILABLE:
            raise ImportError("ebooklib and beautifulsoup4 required for EPUB support")
        
        try:
            book = epub.read_epub(file_path)
            chapters = []
            full_text = ""
            
            # Get book metadata
            title = book.get_metadata('DC', 'title')
            title = title[0][0] if title else "Unknown Title"
            
            # Extract chapters
            chapter_num = 1
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up text
                    text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excessive newlines
                    text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
                    text = text.strip()
                    
                    if text and len(text) > 100:  # Only include substantial content
                        # Try to detect chapter title
                        lines = text.split('\n')
                        chapter_title = None
                        
                        for line in lines[:5]:  # Check first 5 lines
                            line = line.strip()
                            if line and any(re.match(pattern, line, re.IGNORECASE) for pattern in self.chapter_patterns):
                                chapter_title = line
                                break
                        
                        if not chapter_title:
                            # Use item name or generate title
                            item_name = getattr(item, 'file_name', f'Chapter {chapter_num}')
                            chapter_title = f"Chapter {chapter_num}: {item_name.split('/')[-1].replace('.xhtml', '').replace('.html', '')}"
                        
                        chapters.append({
                            'title': chapter_title,
                            'content': text,
                            'word_count': len(text.split()),
                            'char_count': len(text)
                        })
                        
                        full_text += f"\n\n{chapter_title}\n\n{text}"
                        chapter_num += 1
            
            return full_text.strip(), chapters
            
        except Exception as e:
            raise Exception(f"Error reading EPUB file: {str(e)}")
    
    def extract_pdf_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract content from PDF file with basic chapter detection."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 required for PDF support")
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                chapters = []
                current_chapter = ""
                current_title = "Chapter 1"
                chapter_num = 1
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            # Look for chapter breaks
                            lines = page_text.split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and any(re.match(pattern, line, re.IGNORECASE) for pattern in self.chapter_patterns):
                                    # Save previous chapter
                                    if current_chapter.strip():
                                        chapters.append({
                                            'title': current_title,
                                            'content': current_chapter.strip(),
                                            'word_count': len(current_chapter.split()),
                                            'char_count': len(current_chapter)
                                        })
                                    
                                    # Start new chapter
                                    current_title = line
                                    current_chapter = ""
                                    chapter_num += 1
                                else:
                                    current_chapter += line + " "
                            
                            full_text += page_text + "\n"
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                
                # Add final chapter
                if current_chapter.strip():
                    chapters.append({
                        'title': current_title,
                        'content': current_chapter.strip(),
                        'word_count': len(current_chapter.split()),
                        'char_count': len(current_chapter)
                    })
                
                # If no chapters detected, create one big chapter
                if not chapters and full_text.strip():
                    chapters.append({
                        'title': "Full Document",
                        'content': full_text.strip(),
                        'word_count': len(full_text.split()),
                        'char_count': len(full_text)
                    })
                
                return full_text.strip(), chapters
                
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
    
    def extract_text_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract content from plain text file with chapter detection."""
        try:
            encoding = self.detect_encoding(file_path)
            
            # Try multiple encodings if the detected one fails
            encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            full_text = None
            
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as file:
                        full_text = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if full_text is None:
                raise Exception("Could not decode file with any supported encoding")
            
            # Split into chapters based on patterns
            chapters = []
            lines = full_text.split('\n')
            current_chapter = ""
            current_title = "Chapter 1"
            chapter_num = 1
            
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and any(re.match(pattern, line_stripped, re.IGNORECASE) for pattern in self.chapter_patterns):
                    # Save previous chapter
                    if current_chapter.strip():
                        chapters.append({
                            'title': current_title,
                            'content': current_chapter.strip(),
                            'word_count': len(current_chapter.split()),
                            'char_count': len(current_chapter)
                        })
                    
                    # Start new chapter
                    current_title = line_stripped
                    current_chapter = ""
                    chapter_num += 1
                else:
                    current_chapter += line + "\n"
            
            # Add final chapter
            if current_chapter.strip():
                chapters.append({
                    'title': current_title,
                    'content': current_chapter.strip(),
                    'word_count': len(current_chapter.split()),
                    'char_count': len(current_chapter)
                })
            
            # If no chapters detected, create one big chapter
            if not chapters and full_text.strip():
                chapters.append({
                    'title': "Full Text",
                    'content': full_text.strip(),
                    'word_count': len(full_text.split()),
                    'char_count': len(full_text)
                })
            
            return full_text.strip(), chapters
            
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")
    
    def extract_html_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract content from HTML file."""
        if not BS4_AVAILABLE:
            raise ImportError("beautifulsoup4 required for HTML support")
        
        try:
            encoding = self.detect_encoding(file_path)
            
            # Try multiple encodings if the detected one fails
            encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            html_content = None
            
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as file:
                        html_content = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if html_content is None:
                raise Exception("Could not decode file with any supported encoding")
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            full_text = soup.get_text()
            
            # Clean up text
            full_text = re.sub(r'\n\s*\n', '\n\n', full_text)
            full_text = re.sub(r'[ \t]+', ' ', full_text)
            full_text = full_text.strip()
            
            # Try to detect chapters using headings
            chapters = []
            headings = soup.find_all(['h1', 'h2', 'h3'])
            
            if headings:
                # Split content by headings
                current_chapter = ""
                current_title = "Introduction"
                
                for heading in headings:
                    # Save previous chapter
                    if current_chapter.strip():
                        chapters.append({
                            'title': current_title,
                            'content': current_chapter.strip(),
                            'word_count': len(current_chapter.split()),
                            'char_count': len(current_chapter)
                        })
                    
                    # Start new chapter
                    current_title = heading.get_text().strip()
                    current_chapter = ""
                    
                    # Get content until next heading
                    next_sibling = heading.next_sibling
                    while next_sibling and next_sibling.name not in ['h1', 'h2', 'h3']:
                        if hasattr(next_sibling, 'get_text'):
                            current_chapter += next_sibling.get_text() + "\n"
                        elif isinstance(next_sibling, str):
                            current_chapter += next_sibling
                        next_sibling = next_sibling.next_sibling
                
                # Add final chapter
                if current_chapter.strip():
                    chapters.append({
                        'title': current_title,
                        'content': current_chapter.strip(),
                        'word_count': len(current_chapter.split()),
                        'char_count': len(current_chapter)
                    })
            
            # If no chapters detected, use text-based detection
            if not chapters:
                return self.extract_text_content_from_string(full_text)
            
            return full_text, chapters
            
        except Exception as e:
            raise Exception(f"Error reading HTML file: {str(e)}")
    
    def extract_mobi_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract content from MOBI file with basic support."""
        try:
            # Try using proper MOBI library if available
            if MOBI_AVAILABLE:
                try:
                    # This would require a proper MOBI library
                    # For now, fall back to basic extraction
                    pass
                except:
                    pass
            
            # Basic MOBI text extraction
            # MOBI files are complex binary format, try basic text extraction
            with open(file_path, 'rb') as file:
                content = file.read()
            
            # Try to decode as text (MOBI often has readable text sections)
            text_content = ""
            
            # Try different encodings and extraction methods
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    # Look for text patterns in the binary data
                    decoded = content.decode(encoding, errors='ignore')
                    
                    # Extract readable text (filter out binary junk)
                    lines = decoded.split('\n')
                    readable_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        # Keep lines that are mostly readable text
                        if len(line) > 10:
                            # Check if line contains mostly printable characters
                            printable_chars = sum(1 for c in line if c.isprintable() and ord(c) < 127)
                            if len(line) > 0 and printable_chars / len(line) > 0.8:  # 80% printable ASCII
                                # Additional filtering for meaningful content
                                if not re.match(r'^[^a-zA-Z]*$', line):  # Must contain letters
                                    readable_lines.append(line)
                    
                    if readable_lines and len(readable_lines) > 10:  # Need substantial content
                        text_content = '\n'.join(readable_lines)
                        break
                except:
                    continue
            
            if not text_content or len(text_content) < 500:  # Need more substantial content
                # Try alternative extraction method
                try:
                    # Look for HTML-like content in MOBI
                    decoded = content.decode('latin-1', errors='ignore')
                    
                    # Look for HTML tags which are common in MOBI
                    if '<html' in decoded.lower() or '<p>' in decoded.lower():
                        # Try to extract HTML content
                        if BS4_AVAILABLE:
                            soup = BeautifulSoup(decoded, 'html.parser')
                            for script in soup(["script", "style"]):
                                script.decompose()
                            text_content = soup.get_text()
                            text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
                            text_content = re.sub(r'[ \t]+', ' ', text_content)
                            text_content = text_content.strip()
                except:
                    pass
            
            if not text_content or len(text_content) < 200:
                raise Exception("Could not extract sufficient readable text from MOBI file")
            
            # Clean up the extracted text
            text_content = re.sub(r'\x00+', '', text_content)  # Remove null bytes
            text_content = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text_content)  # Remove control chars
            text_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', text_content)  # Normalize line breaks
            text_content = text_content.strip()
            
            if len(text_content) < 200:
                raise Exception("Extracted text too short after cleaning")
            
            # Use text-based chapter detection
            return self.extract_text_content_from_string(text_content)
            
        except Exception as e:
            raise Exception(f"Error reading MOBI file: {str(e)}. MOBI format has limited support - try converting to .epub format for better results.")
    
    def extract_text_content_from_string(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract chapters from a text string."""
        chapters = []
        lines = text.split('\n')
        current_chapter = ""
        current_title = "Chapter 1"
        chapter_num = 1
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and any(re.match(pattern, line_stripped, re.IGNORECASE) for pattern in self.chapter_patterns):
                # Save previous chapter
                if current_chapter.strip():
                    chapters.append({
                        'title': current_title,
                        'content': current_chapter.strip(),
                        'word_count': len(current_chapter.split()),
                        'char_count': len(current_chapter)
                    })
                
                # Start new chapter
                current_title = line_stripped
                current_chapter = ""
                chapter_num += 1
            else:
                current_chapter += line + "\n"
        
        # Add final chapter
        if current_chapter.strip():
            chapters.append({
                'title': current_title,
                'content': current_chapter.strip(),
                'word_count': len(current_chapter.split()),
                'char_count': len(current_chapter)
            })
        
        # If no chapters detected, create one big chapter
        if not chapters and text.strip():
            chapters.append({
                'title': "Full Text",
                'content': text.strip(),
                'word_count': len(text.split()),
                'char_count': len(text)
            })
        
        return text, chapters
    
    def extract_ebook_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Extract content from various eBook formats.
        
        Returns:
            Tuple of (full_text, chapters_list, metadata)
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Extract content based on file type
        if extension == '.epub':
            full_text, chapters = self.extract_epub_content(str(file_path))
        elif extension == '.pdf':
            full_text, chapters = self.extract_pdf_content(str(file_path))
        elif extension in ['.txt']:
            full_text, chapters = self.extract_text_content(str(file_path))
        elif extension in ['.html', '.htm']:
            full_text, chapters = self.extract_html_content(str(file_path))
        elif extension == '.mobi':
            # Try to extract MOBI as text first, fallback to basic text extraction
            try:
                full_text, chapters = self.extract_mobi_content(str(file_path))
            except Exception:
                # Fallback to text extraction
                try:
                    full_text, chapters = self.extract_text_content(str(file_path))
                except Exception:
                    raise ValueError(f"Cannot process .mobi file. Please convert to .epub, .pdf, or .txt first.")
        else:
            # For other unsupported formats, try to read as text
            try:
                full_text, chapters = self.extract_text_content(str(file_path))
            except Exception:
                raise ValueError(f"Cannot process {extension} files yet. Please convert to .epub, .pdf, or .txt first.")
        
        # Generate metadata
        metadata = {
            'title': file_path.stem,
            'format': extension,
            'file_size': file_path.stat().st_size,
            'total_chapters': len(chapters),
            'total_words': sum(ch['word_count'] for ch in chapters),
            'total_characters': sum(ch['char_count'] for ch in chapters),
            'processed_date': datetime.now().isoformat()
        }
        
        return full_text, chapters, metadata
    
    def clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS pronunciation."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common abbreviations for better pronunciation
        text = re.sub(r'\bDr\.', 'Doctor', text)
        text = re.sub(r'\bMr\.', 'Mister', text)
        text = re.sub(r'\bMrs\.', 'Missus', text)
        text = re.sub(r'\bMs\.', 'Miss', text)
        text = re.sub(r'\bProf\.', 'Professor', text)
        text = re.sub(r'\bSt\.', 'Saint', text)
        text = re.sub(r'\bAve\.', 'Avenue', text)
        text = re.sub(r'\bRd\.', 'Road', text)
        text = re.sub(r'\bBlvd\.', 'Boulevard', text)
        
        # Handle numbers and dates better
        text = re.sub(r'\b(\d+)st\b', r'\1 first', text)
        text = re.sub(r'\b(\d+)nd\b', r'\1 second', text)
        text = re.sub(r'\b(\d+)rd\b', r'\1 third', text)
        text = re.sub(r'\b(\d+)th\b', r'\1 th', text)
        
        # Remove or replace problematic characters
        text = re.sub(r'[\u201c\u201d\u201e]', '"', text)  # Normalize quotes
        text = re.sub(r'[\u2018\u2019]', "'", text)  # Normalize apostrophes
        text = re.sub(r'[\u2014\u2013]', '-', text)   # Normalize dashes
        text = re.sub(r'\u2026', '...', text)    # Normalize ellipsis
        
        # Remove excessive punctuation
        text = re.sub(r'\.{4,}', '...', text)
        text = re.sub(r'-{3,}', '---', text)
        
        return text.strip()
    
    def split_text_for_tts(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into TTS-friendly chunks."""
        # Clean text first
        text = self.clean_text_for_tts(text)
        
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed max_length
            if len(current_chunk) + len(sentence) + 2 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Sentence is too long, split by commas or semicolons
                    if len(sentence) > max_length:
                        parts = re.split(r'[,;]+', sentence)
                        for part in parts:
                            part = part.strip()
                            if len(current_chunk) + len(part) + 2 > max_length:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = part
                            else:
                                current_chunk += (", " if current_chunk else "") + part
                    else:
                        current_chunk = sentence
            else:
                current_chunk += (". " if current_chunk else "") + sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def estimate_audio_duration(self, text: str, words_per_minute: int = 150) -> float:
        """Estimate audio duration in minutes."""
        word_count = len(text.split())
        return word_count / words_per_minute
    
    def get_conversion_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about the eBook for conversion preview."""
        try:
            full_text, chapters, metadata = self.extract_ebook_content(file_path)
            
            # Calculate estimated duration
            estimated_duration = self.estimate_audio_duration(full_text)
            
            info = {
                'metadata': metadata,
                'chapters': [
                    {
                        'title': ch['title'],
                        'word_count': ch['word_count'],
                        'estimated_duration': self.estimate_audio_duration(ch['content'])
                    }
                    for ch in chapters
                ],
                'total_estimated_duration': estimated_duration,
                'success': True,
                'error': None
            }
            
            return info
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'metadata': None,
                'chapters': [],
                'total_estimated_duration': 0
            }

def get_supported_formats() -> Dict[str, str]:
    """Get dictionary of supported eBook formats."""
    return EBookConverter.SUPPORTED_FORMATS.copy()

def analyze_ebook(file_path: str) -> Dict[str, Any]:
    """Analyze an eBook file and return conversion information."""
    converter = EBookConverter()
    return converter.get_conversion_info(file_path)

def convert_ebook_to_text_chunks(file_path: str, max_chunk_length: int = 500) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert eBook to text chunks ready for TTS conversion.
    
    Returns:
        Tuple of (text_chunks, metadata)
        text_chunks: List of dicts with 'title', 'content', 'chunk_index', 'chapter_index'
        metadata: Book metadata and conversion info
    """
    converter = EBookConverter()
    
    try:
        full_text, chapters, metadata = converter.extract_ebook_content(file_path)
        
        text_chunks = []
        chunk_index = 0
        
        for chapter_index, chapter in enumerate(chapters):
            chapter_chunks = converter.split_text_for_tts(chapter['content'], max_chunk_length)
            
            for chunk_text in chapter_chunks:
                text_chunks.append({
                    'title': f"{chapter['title']} - Part {len([c for c in text_chunks if c['chapter_index'] == chapter_index]) + 1}",
                    'content': chunk_text,
                    'chunk_index': chunk_index,
                    'chapter_index': chapter_index,
                    'chapter_title': chapter['title'],
                    'word_count': len(chunk_text.split()),
                    'estimated_duration': converter.estimate_audio_duration(chunk_text)
                })
                chunk_index += 1
        
        return text_chunks, metadata
        
    except Exception as e:
        raise Exception(f"Error converting eBook: {str(e)}") 