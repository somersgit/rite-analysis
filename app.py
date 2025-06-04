import os
from flask import Flask, render_template, request, jsonify
import PyPDF2
import tabula
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import re
from werkzeug.exceptions import RequestEntityTooLarge
import json
import hashlib
from pathlib import Path

# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    load_dotenv()

# Ensure required directories exist
for directory in ['uploads', 'cache']:
    os.makedirs(directory, exist_ok=True)

# Cache configuration
CACHE_DIR = Path('cache')
SUMMARIES_CACHE_FILE = CACHE_DIR / 'summaries_cache.json'

def get_pdf_fingerprint(pdf_path):
    """Generate a fingerprint for the first page of the PDF."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if len(pdf_reader.pages) > 0:
                first_page = pdf_reader.pages[0].extract_text()
                # Create a hash of the first page content
                return hashlib.md5(first_page.encode()).hexdigest()
    except Exception as e:
        print(f"Warning: Could not generate PDF fingerprint: {str(e)}")
    return None

def load_cache():
    """Load the summaries cache from file."""
    if SUMMARIES_CACHE_FILE.exists():
        try:
            with open(SUMMARIES_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache: {str(e)}")
    return {}

def save_cache(cache_data):
    """Save the summaries cache to file."""
    try:
        with open(SUMMARIES_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Could not save cache: {str(e)}")

# Initialize cache
summaries_cache = load_cache()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def clean_text(text):
    """Clean up text by fixing common PDF formatting issues."""
    # Fix common spacing issues
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between lower and uppercase letters
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
    text = re.sub(r'([a-z])-\s+([a-z])', r'\1\2', text)  # Fix hyphenation
    # Fix specific PDF formatting issues
    text = text.replace('a ttacks', 'attacks')  # Common PDF error
    text = text.replace('ADUL T', 'ADULT')  # Fix known typo
    return text.strip()

def extract_wrong_answer_rates(pdf_path):
    # Read the PDF using PyPDF2
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    # Define patterns to match question numbers and percentages
    # This pattern looks for:
    # 1. 1-3 digits (question number)
    # 2. Followed by optional whitespace
    # 3. Followed by either:
    #    - A percentage (1-3 digits followed by %)
    #    - The word "percent" or "%"
    patterns = [
        r"(\d{1,3})\s*(\d{1,3})%",  # Matches "371 60%"
        r"(\d{1,3})\s+(\d{1,3})\s*percent",  # Matches "371 60 percent"
        r"Question\s+(\d{1,3})[^\d]*?(\d{1,3})%"  # Matches "Question 371...60%"
    ]
    
    high_error_questions = {'60-79': [], '80+': []}
    found_questions = set()  # To avoid duplicates
    
    # Process each pattern
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                question_num = int(match.group(1))
                percentage = int(match.group(2))
                
                # Skip if we've already processed this question
                if question_num in found_questions:
                    continue
                
                if 60 <= percentage < 80:
                    high_error_questions['60-79'].append(question_num)
                    found_questions.add(question_num)
                elif percentage >= 80:
                    high_error_questions['80+'].append(question_num)
                    found_questions.add(question_num)
            except (ValueError, IndexError) as e:
                print(f"Error processing match {match.group()}: {str(e)}")
                continue
    
    # Sort the question numbers for better presentation
    high_error_questions['60-79'].sort()
    high_error_questions['80+'].sort()
    
    # Print debug information
    print(f"\nFound {len(high_error_questions['60-79'])} questions with 60-79% wrong answers:")
    print(high_error_questions['60-79'])
    print(f"\nFound {len(high_error_questions['80+'])} questions with 80%+ wrong answers:")
    print(high_error_questions['80+'])
    print(f"\nTotal questions found: {len(found_questions)}")
    
    return high_error_questions

def extract_page_headers(pdf_reader):
    """Extract headers from each page and map them to page numbers and track questions."""
    page_categories = {}
    category_questions = {}  # Track questions for each category
    current_category = None
    full_text = ""
    
    # First, get full text with page markers
    for page_num in range(len(pdf_reader.pages)):
        try:
            page_text = pdf_reader.pages[page_num].extract_text()
            if page_text:
                # Add page marker with the actual page text
                full_text += f"\n[PAGE {page_num + 1}]\n{page_text}"
        except Exception as e:
            print(f"Error extracting text from page {page_num}: {str(e)}")
    
    # Process pages to identify categories and their boundaries
    current_page = 0
    current_category = None
    
    # Split text into pages
    pages = full_text.split('[PAGE')
    
    for page in pages[1:]:  # Skip first empty split
        try:
            # Extract page number
            page_num_match = re.match(r'\s*(\d+)\s*\]', page)
            if page_num_match:
                page_num = int(page_num_match.group(1)) - 1
                page_content = page[page_num_match.end():]
                
                # Skip first two pages
                if page_num < 2:
                    page_categories[page_num] = "Uncategorized"
                    continue
                
                # Look for category in first few lines
                lines = [line.strip() for line in page_content.split('\n') if line.strip()]
                category_found = False
                
                for line in lines[:3]:
                    # Skip lines that are just page numbers
                    if line.isdigit():
                        continue
                    
                    # Look for header pattern: text followed by optional page number
                    header_match = re.search(r'^(.*?)(?:\s+\d+\s*)?$', line)
                    if header_match:
                        potential_category = header_match.group(1).strip()
                        
                        # Skip if it's just a number or too short
                        if potential_category.isdigit() or len(potential_category) < 5:
                            continue
                        
                        # Remove any parenthetical content and clean up
                        clean_category = re.sub(r'\s*\(.*?\)\s*', '', potential_category)
                        clean_category = re.sub(r'^\d+\.\s*', '', clean_category)  # Remove leading numbers
                        clean_category = clean_category.strip()
                        
                        if clean_category:
                            current_category = clean_category
                            if current_category not in category_questions:
                                category_questions[current_category] = set()
                            category_found = True
                            break
                
                # Assign category to page
                if category_found:
                    page_categories[page_num] = current_category
                elif current_category and page_num > 0:
                    page_categories[page_num] = current_category
                else:
                    page_categories[page_num] = "Uncategorized"
                
                # Find questions on this page
                if current_category and current_category != "Uncategorized":
                    # Look for lines that start with a number and contain all caps text
                    lines = page_content.split('\n')
                    for line in lines:
                        # Look for pattern: number followed by all caps text
                        match = re.match(r'^\s*(\d+)\s+(.+)$', line)
                        if match:
                            try:
                                question_num = int(match.group(1))
                                remaining_text = match.group(2).strip()
                                # Verify it's a reasonable question number and all remaining text is caps
                                if 1 <= question_num <= 1000 and remaining_text.isupper():
                                    category_questions[current_category].add(question_num)
                            except (ValueError, IndexError):
                                continue
        
        except Exception as e:
            print(f"Error processing page: {str(e)}")
            continue
    
    # Print debug information
    print("\nQuestion counts per category:")
    for category, questions in category_questions.items():
        print(f"{category}: {len(questions)} questions - {sorted(list(questions))}")
    
    # Convert sets to counts
    category_question_counts = {cat: len(questions) for cat, questions in category_questions.items()}
    
    return page_categories, category_question_counts, category_questions

def extract_general_category(pdf_reader, page_num):
    """Extract the general category for a specific page."""
    # This function now just returns the category from the page_categories mapping
    # The actual extraction is done once by extract_page_headers
    return "Uncategorized"  # This will be replaced by the mapping from extract_page_headers

def extract_question_info(pdf_path, question_numbers):
    # Initialize the OpenAI client
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url="https://api.openai.com/v1"
    )
    question_info = {}
    
    # Read the PDF and extract page categories first
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        page_categories, category_question_counts, category_questions = extract_page_headers(pdf_reader)
        print("\nExtracted page categories:")
        for page_num, category in sorted(page_categories.items()):
            print(f"Page {page_num + 1}: {category}")
    
    # Check if we have processed this PDF before
    pdf_fingerprint = get_pdf_fingerprint(pdf_path)
    cache_hit = False
    
    if pdf_fingerprint and pdf_fingerprint in summaries_cache:
        print("\nFound cached summaries for this PDF!")
        cached_data = summaries_cache[pdf_fingerprint]
        # Check if all requested questions are in cache
        all_cached = all(str(q_num) in cached_data for q_num in question_numbers)
        
        if all_cached:
            print("Using cached summaries for all questions")
            # Create a new dictionary with updated general categories
            updated_cache = {}
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page_text = pdf_reader.pages[page_num].extract_text()
                        if page_text:
                            full_text += f"\n[PAGE {page_num + 1}]\n" + page_text
                    except Exception as e:
                        continue
                
                for q_num in question_numbers:
                    cached_question = cached_data[str(q_num)]
                    if isinstance(cached_question, dict):
                        # Find which page this question is on
                        current_page = 0
                        question_pattern = f"Question #{q_num}|\\n{q_num}\\s+[A-Z]"
                        matches = list(re.finditer(question_pattern, full_text))
                        if matches:
                            pos = matches[0].start()
                            page_markers = list(re.finditer(r'\[PAGE (\d+)\]', full_text[:pos]))
                            if page_markers:
                                current_page = int(page_markers[-1].group(1)) - 1
                        
                        # Add or update general category
                        updated_question = dict(cached_question)
                        updated_question['general_category'] = page_categories.get(current_page, "Uncategorized")
                        updated_cache[q_num] = updated_question
                    else:
                        updated_cache[q_num] = {
                            'category': 'Error',
                            'subcategory': None,
                            'general_category': 'Error',
                            'content': cached_question,
                            'reference': None,
                            'summary': cached_question
                        }
            
            return updated_cache
        else:
            print("Some questions not found in cache, will process missing questions")
            # Use cached data for questions we have
            question_info = {}
            for q_num in question_numbers:
                if str(q_num) in cached_data:
                    cached_question = cached_data[str(q_num)]
                    if isinstance(cached_question, dict):
                        updated_question = dict(cached_question)
                        # We'll update the general category later when processing the full text
                        question_info[q_num] = updated_question
            
            # Only process questions we don't have
            question_numbers = [q_num for q_num in question_numbers if str(q_num) not in cached_data]
    
    # Read the manual PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        full_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            try:
                page_text = pdf_reader.pages[page_num].extract_text()
                if page_text:
                    full_text += f"\n[PAGE {page_num + 1}]\n" + page_text
            except Exception as e:
                print(f"Warning: Error extracting text from page {page_num + 1}: {str(e)}")
                continue
    
    if not full_text.strip():
        raise Exception("Failed to extract any text from the PDF")
    
    # Process each question that wasn't in cache
    newly_processed = {}
    for q_num in question_numbers:
        try:
            # More specific pattern matching for question numbers
            patterns = [
                f"Question #{q_num}\\s+[A-Z]",
                f"\\n{q_num}\\s+[A-Z]",
                f"^{q_num}\\s+[A-Z]",
                f"[^0-9]{q_num}\\s+[A-Z]"
            ]
            
            # Find the first matching pattern
            start_idx = -1
            matched_text = ""
            current_page = 0
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, full_text))
                if matches:
                    for match in matches:
                        pos = match.start()
                        # Find which page this question is on
                        page_markers = list(re.finditer(r'\[PAGE (\d+)\]', full_text[:pos]))
                        if page_markers:
                            current_page = int(page_markers[-1].group(1)) - 1
                        
                        next_q = re.search(r'Question #\d{1,3}|\\n\d{1,3}\\s+[A-Z]', full_text[pos+1:pos+500])
                        if not next_q or (next_q and int(re.search(r'\d+', next_q.group()).group()) > q_num):
                            start_idx = pos
                            matched_text = match.group()
                            break
                    if start_idx != -1:
                        break
            
            if start_idx == -1:
                print(f"Warning: Question {q_num} not found with primary patterns, trying fallback patterns")
                fallback_patterns = [
                    f"{q_num}\\s+",
                    f"Question\\s+{q_num}",
                    f"\\n{q_num}\\s+"
                ]
                for pattern in fallback_patterns:
                    matches = list(re.finditer(pattern, full_text))
                    if matches:
                        start_idx = matches[0].start()
                        matched_text = matches[0].group()
                        # Find which page this question is on
                        page_markers = list(re.finditer(r'\[PAGE (\d+)\]', full_text[:start_idx]))
                        if page_markers:
                            current_page = int(page_markers[-1].group(1)) - 1
                        break
            
            if start_idx == -1:
                print(f"Warning: Could not find content for Question {q_num}")
                question_info[q_num] = f"Question {q_num} information not found"
                continue
            
            # Get the general category for this question from the page it's on
            general_category = page_categories.get(current_page, "Uncategorized")
            print(f"Question {q_num} found on page {current_page + 1}, category: {general_category}")
            
            # Rest of your existing question processing code...
            next_q_pattern = r'Question #\d{1,3}|\n\d{1,3}\s+[A-Z]|\[PAGE \d+\]'
            next_q_match = re.search(next_q_pattern, full_text[start_idx + len(matched_text):])
            
            if next_q_match:
                end_idx = start_idx + len(matched_text) + next_q_match.start()
            else:
                end_idx = len(full_text)
            
            question_text = full_text[start_idx:end_idx].strip()
            question_text = re.sub(r'\[PAGE \d+\]\s*', ' ', question_text)
            lines = [line.strip() for line in question_text.split('\n') if line.strip()]
            
            if not lines:
                print(f"Warning: No content found for Question {q_num}")
                question_info[q_num] = f"No content found for Question {q_num}"
                continue
            
            first_line = lines[0]
            category_text = re.sub(f'^(?:Question\\s+#{q_num}|{q_num})\\s*', '', first_line).strip()
            parts = [p.strip() for p in re.split(r'\s{2,}', category_text) if p.strip()]
            
            category_parts = []
            subcategory = None
            
            for part in parts:
                cleaned_part = clean_text(part)
                if cleaned_part.isupper() or any(phrase in cleaned_part for phrase in ["CORE KNOWLEDGE"]):
                    category_parts.append(cleaned_part)
                elif not subcategory and cleaned_part[0].isupper():
                    subcategory = cleaned_part
            
            category = " ".join(category_parts) if category_parts else "Category Not Found"
            
            if not subcategory and len(lines) > 1:
                second_line = clean_text(lines[1])
                if not second_line.isupper() and second_line[0].isupper():
                    subcategory = second_line
            
            content_start = 2 if subcategory in lines[1:2] else 1
            content = " ".join(lines[content_start:])
            content = clean_text(content)
            
            if not content:
                print(f"Warning: No content extracted for Question {q_num}")
                content = "Content not found"
            
            reference = None
            ref_match = re.search(r'(?:Reference|References):\s*([^\n]+)', content, re.IGNORECASE)
            if ref_match:
                reference = clean_text(ref_match.group(1))
                content = clean_text(content[:ref_match.start()].strip())
            
            print(f"\nProcessing Question {q_num}:")
            print(f"Category: {category}")
            print(f"Subcategory: {subcategory}")
            print(f"General Category: {general_category}")
            print(f"Content length: {len(content)} characters")
            
            prompt = (
                f"You are analyzing a medical examination question. Based on the following information, provide a concise summary:\n\n"
                f"Question Number: {q_num}\n"
                f"Category: {category}\n"
                f"Subcategory: {subcategory}\n"
                f"General Category: {general_category}\n"
                f"Content: {content}\n\n"
                f"Please provide a brief, specific summary that covers:\n"
                f"1. The exact medical knowledge or concept being tested\n"
                f"2. Why this specific topic is important for medical residents\n"
                f"Keep the summary focused and under 100 words."
            )
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=150
                )
                
                summary = response.choices[0].message.content
            except Exception as e:
                print(f"Warning: Error generating summary for Question {q_num}: {str(e)}")
                summary = "Error generating summary"
            
            result = {
                'category': category,
                'subcategory': subcategory,
                'general_category': general_category,
                'content': content,
                'reference': reference,
                'summary': summary
            }
            
            question_info[q_num] = result
            newly_processed[str(q_num)] = result
            
        except Exception as e:
            print(f"Error processing question {q_num}: {str(e)}")
            error_result = {
                'category': 'Error',
                'subcategory': None,
                'general_category': 'Error',
                'content': f"Error processing question: {str(e)}",
                'reference': None,
                'summary': 'Error processing question'
            }
            question_info[q_num] = error_result
            newly_processed[str(q_num)] = error_result
    
    if pdf_fingerprint and newly_processed:
        if pdf_fingerprint not in summaries_cache:
            summaries_cache[pdf_fingerprint] = {}
        summaries_cache[pdf_fingerprint].update(newly_processed)
        save_cache(summaries_cache)
        print(f"\nCached {len(newly_processed)} new question summaries")
    
    return question_info

def generate_teaching_points(question_info_60_79, question_info_80_plus):
    """Generate key teaching points based on question summaries."""
    # Combine all summaries
    all_summaries = []
    for questions in [question_info_60_79, question_info_80_plus]:
        for info in questions.values():
            if isinstance(info, dict) and 'summary' in info:
                all_summaries.append(info['summary'])
    
    if not all_summaries:
        return []

    # Initialize the OpenAI client
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url="https://api.openai.com/v1"
    )
    
    prompt = f"""As a chief resident, analyze these question summaries from commonly missed RITE exam questions and provide key teaching points. Focus on:
1. Common themes and patterns
2. Critical knowledge gaps
3. High-yield topics for resident education
4. Practical teaching strategies

Summaries to analyze:
{' '.join(all_summaries)}

Provide a concise, bullet-pointed list of 5-7 key teaching points that would be most valuable for chief residents."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        teaching_points = response.choices[0].message.content.strip()
        return teaching_points
    except Exception as e:
        print(f"Error generating teaching points: {str(e)}")
        return "Error generating teaching points. Please try again."

@app.route('/')
def index():
    empty_result = {
        'stats': {
            'category_summary': {
                'total_pages': 0,
                'uncategorized_pages': 0,
                'category_frequency': {},
                'category_questions': {},
                'high_error_counts': {}
            }
        }
    }
    return render_template('index.html', result=empty_result)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'wrong_rates_pdf' not in request.files or 'manual_pdf' not in request.files:
        return jsonify({'error': 'Both PDF files are required'}), 400
    
    wrong_rates_file = request.files['wrong_rates_pdf']
    manual_file = request.files['manual_pdf']
    
    wrong_rates_path = os.path.join(app.config['UPLOAD_FOLDER'], 'wrong_rates.pdf')
    manual_path = os.path.join(app.config['UPLOAD_FOLDER'], 'manual.pdf')
    
    wrong_rates_file.save(wrong_rates_path)
    manual_file.save(manual_path)
    
    try:
        # First, get all page categories and question mappings from the manual
        with open(manual_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_categories, category_question_counts, category_questions = extract_page_headers(pdf_reader)
        
        # Get questions with high error rates
        high_error_questions = extract_wrong_answer_rates(wrong_rates_path)
        all_questions = high_error_questions['60-79'] + high_error_questions['80+']
        
        # Get detailed question info
        question_info = extract_question_info(manual_path, all_questions)
        
        questions_60_79 = {q: question_info[q] for q in high_error_questions['60-79']}
        questions_80_plus = {q: question_info[q] for q in high_error_questions['80+']}
        
        teaching_points = generate_teaching_points(questions_60_79, questions_80_plus)
        
        # Prepare statistics
        subcategory_stats = {'60-79': {}, '80+': {}}
        general_category_stats = {'60-79': {}, '80+': {}}
        population_stats = {'60-79': {}, '80+': {}}
        
        # Calculate category statistics
        category_summary = {
            'total_pages': len(page_categories),
            'uncategorized_pages': sum(1 for cat in page_categories.values() if cat == "Uncategorized"),
            'category_frequency': {},
            'high_error_counts': {}  # Add tracking for high-error questions per category
        }
        
        # Count frequency of each category and initialize high-error counts
        for category in page_categories.values():
            if category != "Uncategorized":
                category_summary['category_frequency'][category] = category_summary['category_frequency'].get(category, 0) + 1
                category_summary['high_error_counts'][category] = {
                    '60-79': 0,
                    '80+': 0,
                    'total': 0
                }
        
        # Add question counts to category summary
        category_summary['category_questions'] = category_question_counts
        
        # Count high-error questions per category using the passed category_questions
        for error_range, questions in high_error_questions.items():
            for q_num in questions:
                # Find which category this question belongs to
                for category, q_set in category_questions.items():
                    if q_num in q_set:
                        category_summary['high_error_counts'][category][error_range] += 1
                        category_summary['high_error_counts'][category]['total'] += 1
                        break
        
        # Sort categories by frequency
        category_summary['category_frequency'] = dict(
            sorted(category_summary['category_frequency'].items(), 
                  key=lambda x: x[1], 
                  reverse=True)
        )
        
        # Process categories for 60-79% questions
        for q_num in high_error_questions['60-79']:
            if isinstance(question_info[q_num], dict):
                # Subcategory statistics
                subcategory = question_info[q_num]['subcategory']
                if subcategory:
                    subcategory_stats['60-79'][subcategory] = subcategory_stats['60-79'].get(subcategory, 0) + 1
                
                # General category statistics
                general_category = question_info[q_num]['general_category']
                if general_category:
                    general_category_stats['60-79'][general_category] = general_category_stats['60-79'].get(general_category, 0) + 1
                
                # Population statistics
                category = question_info[q_num]['category']
                pop_type = 'Adult' if 'ADULT' in category else 'Pediatric' if 'PEDIATRIC' in category else 'Not Specified'
                population_stats['60-79'][pop_type] = population_stats['60-79'].get(pop_type, 0) + 1
        
        # Process categories for 80+% questions
        for q_num in high_error_questions['80+']:
            if isinstance(question_info[q_num], dict):
                # Subcategory statistics
                subcategory = question_info[q_num]['subcategory']
                if subcategory:
                    subcategory_stats['80+'][subcategory] = subcategory_stats['80+'].get(subcategory, 0) + 1
                
                # General category statistics
                general_category = question_info[q_num]['general_category']
                if general_category:
                    general_category_stats['80+'][general_category] = general_category_stats['80+'].get(general_category, 0) + 1
                
                # Population statistics
                category = question_info[q_num]['category']
                pop_type = 'Adult' if 'ADULT' in category else 'Pediatric' if 'PEDIATRIC' in category else 'Not Specified'
                population_stats['80+'][pop_type] = population_stats['80+'].get(pop_type, 0) + 1
        
        result = {
            'stats': {
                '60-79': len(high_error_questions['60-79']),
                '80+': len(high_error_questions['80+']),
                'categories': subcategory_stats,
                'general_categories': general_category_stats,
                'population': population_stats,
                'category_summary': category_summary
            },
            'teaching_points': teaching_points,
            'questions_60_79': questions_60_79,
            'questions_80_plus': questions_80_plus
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(wrong_rates_path):
            os.remove(wrong_rates_path)
        if os.path.exists(manual_path):
            os.remove(manual_path)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        'error': 'File too large',
        'message': 'The uploaded file exceeds the maximum allowed size of 64MB. Please try with a smaller file.'
    }), 413

if __name__ == '__main__':
    # Use environment variable for port with a default of 5000
    port = int(os.environ.get('PORT', 5000))
    # In development, use debug mode. In production, use host='0.0.0.0'
    if os.environ.get('RENDER'):
        app.run(host='0.0.0.0', port=port)
    else:
        app.run(debug=True, port=port) 