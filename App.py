# app.py
import streamlit as st
import re
import random
import string
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import os
import hashlib

# Download required NLTK data (will be cached)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()

class AIHumanizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.synonym_cache = {}
        
    def get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to first character lemmatize() accepts"""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                   "N": wordnet.NOUN,
                   "V": wordnet.VERB,
                   "R": wordnet.ADV}
        return tag(dict.get(tag_dict, wordnet.NOUN))
    
    def get_synonyms(self, word: str, pos: str = None) -> List[str]:
        """Get synonyms for a word using WordNet"""
        cache_key = f"{word}_{pos}"
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]
            
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                # Only add if it's a single word and different from original
                if synonym.lower() != word.lower() and ' ' not in synonym:
                    synonyms.add(synonym.lower())
        
        result = list(synonyms)[:5]  # Limit to 5 synonyms
        self.synonym_cache[cache_key] = result
        return result
    
    def paraphrase_sentence(self, sentence: str) -> str:
        """Paraphrase a single sentence using multiple techniques"""
        if len(sentence.strip()) < 10:
            return sentence
            
        words = word_tokenize(sentence)
        if len(words) < 3:
            return sentence
            
        # POS tagging
        pos_tags = pos_tag(words)
        
        # Build paraphrased sentence
        paraphrased_words = []
        i = 0
        while i < len(words):
            word = words[i]
            pos = pos_tags[i][1] if i < len(pos_tags) else 'NN'
            
            # Skip punctuation and very short words
            if word.lower() in string.punctuation or len(word) <= 2:
                paraphrased_words.append(word)
                i += 1
                continue
                
            # Occasionally add filler words or change structure
            if random.random() < 0.15 and len(paraphrased_words) > 0:
                fillers = ['actually', 'basically', 'essentially', 'in fact', 'indeed', 
                          'specifically', 'particularly', 'generally', 'typically']
                if paraphrased_words[-1] not in string.punctuation:
                    paraphrased_words.append(random.choice(fillers))
            
            # Synonym replacement (not for stopwords)
            if word.lower() not in self.stop_words and random.random() < 0.3:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    paraphrased_words.append(random.choice(synonyms))
                else:
                    paraphrased_words.append(word)
            else:
                paraphrased_words.append(word)
            
            # Occasionally add discourse markers
            if random.random() < 0.1 and i < len(words) - 1:
                markers = [', however,', ', therefore,', ', consequently,', 
                          ', moreover,', ', furthermore,', ', meanwhile,']
                paraphrased_words.append(random.choice(markers))
            
            i += 1
        
        # Reconstruct sentence
        paraphrased = ' '.join(paraphrased_words)
        
        # Clean up multiple spaces and punctuation
        paraphrased = re.sub(r'\s+', ' ', paraphrased)
        paraphrased = re.sub(r'\s([,.!?;:])', r'\1', paraphrased)
        
        return paraphrased.strip()
    
    def vary_sentence_structure(self, sentences: List[str]) -> List[str]:
        """Apply various sentence structure modifications"""
        varied_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Skip empty sentences
            if not sentence.strip():
                varied_sentences.append(sentence)
                continue
                
            # Add variety to sentence beginnings
            if i > 0 and random.random() < 0.4:
                beginnings = [
                    'Additionally, ', 'Moreover, ', 'Furthermore, ', 
                    'In addition, ', 'Also, ', 'Besides, ',
                    'On the other hand, ', 'However, ', 'Nevertheless, ',
                    'Consequently, ', 'Therefore, ', 'Thus, '
                ]
                
                # Don't add if sentence already starts with transition word
                first_word = sentence.split()[0].lower() if sentence.split() else ''
                transition_words = {'however', 'therefore', 'thus', 'consequently', 
                                  'moreover', 'furthermore', 'additionally', 'also'}
                
                if first_word not in transition_words:
                    sentence = random.choice(beginnings) + sentence.lower().capitalize()
            
            # Occasionally combine sentences (not too often to maintain readability)
            if (i < len(sentences) - 1 and len(sentence.split()) < 15 and 
                len(sentences[i+1].split()) < 15 and random.random() < 0.2):
                combined = sentence + ' ' + sentences[i+1].lower()
                varied_sentences.append(combined)
                # Skip next sentence since we combined it
                continue
            
            varied_sentences.append(sentence)
            
        return varied_sentences
    
    def add_human_elements(self, text: str) -> str:
        """Add human-like imperfections and variations"""
        # Add occasional typos (very rare - 1 in 500 words)
        words = text.split()
        if len(words) > 20 and random.random() < 0.8:  # 80% chance to add human elements
            # Add sentence fragments occasionally
            if random.random() < 0.1:
                fragments = ['you know', 'I mean', 'sort of', 'kind of', 'like', 'well']
                insert_pos = random.randint(1, max(1, len(words) - 2))
                words.insert(insert_pos, f", {random.choice(fragments)}")
            
            # Add slight redundancy (humans repeat concepts)
            if random.random() < 0.15 and len(words) > 50:
                # Find a key concept and rephrase it later
                key_words = [w for w in words if len(w) > 5 and w.lower() not in self.stop_words]
                if key_words:
                    concept = random.choice(key_words)
                    synonyms = self.get_synonyms(concept)
                    if synonyms:
                        insert_pos = min(len(words), random.randint(len(words)//2, len(words)-1))
                        if insert_pos < len(words):
                            words.insert(insert_pos, f"(or {random.choice(synonyms)})")
        
        return ' '.join(words)
    
    def humanize_text(self, text: str) -> str:
        """Main method to humanize AI-generated text"""
        if not text.strip():
            return text
            
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Paraphrase each sentence
        paraphrased_sentences = []
        for sentence in sentences:
            paraphrased = self.paraphrase_sentence(sentence)
            paraphrased_sentences.append(paraphrased)
        
        # Vary sentence structure
        varied_sentences = self.vary_sentence_structure(paraphrased_sentences)
        
        # Join back into text
        humanized_text = ' '.join(varied_sentences)
        
        # Add human elements
        humanized_text = self.add_human_elements(humanized_text)
        
        # Final cleanup
        humanized_text = re.sub(r'\s+', ' ', humanized_text)
        humanized_text = humanized_text.strip()
        
        return humanized_text

def main():
    st.set_page_config(
        page_title="AI Humanizer - Make AI Content Undetectable",
        page_icon="🎭",
        layout="wide"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .result-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-header'>🎭 AI Humanizer</h1>", unsafe_allow_html=True)
    
    # Warning disclaimer
    st.markdown("""
    <div class="warning-box">
    <strong>Important Disclaimer:</strong> This tool uses local processing only and cannot guarantee 
    complete undetectability by advanced AI detection systems like Turnitin. AI detectors continuously 
    evolve, and no method can provide 100% assurance against detection. Use responsibly and ethically.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize humanizer
    humanizer = AIHumanizer()
    
    # Input section
    st.subheader("Enter AI-Generated Content")
    input_text = st.text_area(
        "Paste your AI-generated text below:",
        height=200,
        placeholder="Enter the text you want to humanize...",
        key="input_text"
    )
    
    # Process button
    if st.button("🎭 Humanize Text", type="primary", use_container_width=True):
        if not input_text.strip():
            st.error("Please enter some text to humanize!")
        else:
            with st.spinner("Humanizing your text... This may take a moment."):
                try:
                    # Process the text
                    humanized_result = humanizer.humanize_text(input_text)
                    
                    # Display results
                    st.subheader("Humanized Result")
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.text_area(
                        "Humanized Text:",
                        value=humanized_result,
                        height=250,
                        key="output_text"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add copy button functionality
                    st.download_button(
                        label="💾 Download Humanized Text",
                        data=humanized_result,
                        file_name="humanized_text.txt",
                        mime="text/plain"
                    )
                    
                    # Statistics
                    original_words = len(input_text.split())
                    humanized_words = len(humanized_result.split())
                    st.info(f"Original: {original_words} words | Humanized: {humanized_words} words")
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
    
    # Information section
    with st.expander("ℹ️ How This Works"):
        st.markdown("""
        ### Local AI Humanization Process
        
        This application uses **100% local processing** with no external APIs:
        
        1. **Synonym Replacement**: Uses NLTK's WordNet to replace words with appropriate synonyms
        2. **Sentence Restructuring**: Varies sentence beginnings and occasionally combines sentences
        3. **Human-like Elements**: Adds natural discourse markers, occasional redundancy, and human speech patterns
        4. **Variability**: Each run produces slightly different results due to randomization
        
        ### Limitations
        
        - **No Guarantee of Undetectability**: Advanced AI detectors use sophisticated methods that may still identify patterns
        - **Local Processing Only**: Quality is limited compared to cloud-based AI services
        - **Best for Minor Modifications**: Works best for light humanization rather than complete content transformation
        
        ### Ethical Use
        
        Please use this tool responsibly and in accordance with your institution's policies on AI usage.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "AI Humanizer v1.0 • All processing done locally • No data is stored or transmitted"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
