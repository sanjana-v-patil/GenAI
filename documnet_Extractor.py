class DocumentExtractorAgent(Agent):
    """Agent responsible for extracting text from documents"""
    def __init__(self):
        super().__init__("DocumentExtractor")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        self.logger.info(f"Extracting text from {pdf_path}")
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text("text") + "\n" for page in doc)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process(self, pdf_files: List[str]) -> Dict[str, str]:
        """Process a list of PDF files and extract text"""
        documents = {}
        for pdf in pdf_files:
            try:
                documents[os.path.basename(pdf)] = self.extract_text_from_pdf(pdf)
                self.logger.info(f"Successfully extracted text from {pdf}")
            except Exception as e:
                self.logger.error(f"Failed to extract text from {pdf}: {e}")
        return documents
