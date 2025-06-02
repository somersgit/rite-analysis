# RITE Exam Analysis Tool

A web application for analyzing RITE exam results, providing detailed visualizations and teaching points.

## Features

- PDF text extraction and analysis
- Interactive visualizations using Chart.js
- Question categorization by topic, general category, and population type
- Caching system for improved performance
- Teaching points generation

## Local Development Setup

1. Clone the repository
2. Install Python 3.8 or higher
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
5. Run the application:
   ```bash
   python app.py
   ```
6. Visit `http://localhost:5000` in your browser

## Deployment on Render

1. Fork this repository to your GitHub account
2. Create a new Web Service on [Render](https://render.com)
3. Connect your GitHub repository
4. Add the following environment variable:
   - `OPENAI_API_KEY`: Your OpenAI API key
5. Deploy! Render will automatically use the configuration in `render.yaml`

## Directory Structure

- `/uploads`: Temporary storage for uploaded PDFs
- `/cache`: Cache directory for processed results
- `/static`: Static assets (CSS, JS, images)
- `/templates`: HTML templates

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## Notes

- The application uses caching to improve performance
- Uploaded PDFs are temporarily stored and processed
- Results are cached using MD5 hashes for efficiency

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Modern web browser
- Java Runtime Environment (JRE) for PDF processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd riteAnalysis
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload two PDF files:
   - Wrong Answer Rates PDF: Contains the percentage of wrong answers for each question
   - Question Manual PDF: Contains the actual questions and their categories

4. The application will process the PDFs and display:
   - Summary statistics
   - Interactive visualizations
   - Detailed question analysis
   - AI-generated teaching points

## File Structure

- `app.py`: Main Flask application
- `templates/index.html`: Frontend interface
- `uploads/`: Temporary storage for uploaded files
- `cache/`: Cached analysis results
- `.env`: Environment variables (API key)

## Security Notes

- Never commit your `.env` file or expose your API key
- The application automatically cleans up uploaded files
- Uses secure file handling practices
- Implements file size limits and type checking

## Caching

The application caches processed questions to improve performance and reduce API costs. Cached results are stored in:
```
cache/summaries_cache.json
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license]

## Acknowledgments

- OpenAI for GPT API
- Flask framework
- Chart.js for visualizations 