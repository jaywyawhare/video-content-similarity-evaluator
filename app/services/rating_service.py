from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
from typing import Dict, Set, Union, Optional
import re
from ..core.logger import get_logger
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
import string
from collections import Counter

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    nltk.download("stopwords")
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")


class RatingService:
    def __init__(self):
        # Add more filler words from POC
        self.filler_words: Set[str] = {
            "absolutely",
            "actual",
            "actually",
            "alright",
            "amazing",
            "anyway",
            "apparently",
            "approximately",
            "badly",
            "basically",
            "begin",
            "certainly",
            "clearly",
            "completely",
            "definitely",
            "easily",
            "effectively",
            "entirely",
            "especially",
            "essentially",
            "exactly",
            "extremely",
            "fairly",
            "frankly",
            "frequently",
            "fully",
            "generally",
            "hardly",
            "heavily",
            "highly",
            "hmm",
            "honestly",
            "hopefully",
            "just",
            "largely",
            "like",
            "literally",
            "maybe",
            "might",
            "most",
            "mostly",
            "much",
            "necessarily",
            "nicely",
            "obviously",
            "ok",
            "okay",
            "particularly",
            "perhaps",
            "possibly",
            "practically",
            "precisely",
            "primarily",
            "probably",
            "quite",
            "rather",
            "real",
            "really",
            "relatively",
            "right",
            "seriously",
            "significantly",
            "simply",
            "slightly",
            "so",
            "specifically",
            "start",
            "strongly",
            "stuff",
            "surely",
            "things",
            "too",
            "totally",
            "truly",
            "try",
            "typically",
            "uh",
            "ultimately",
            "um",
            "usually",
            "very",
            "virtually",
            "well",
            "whatever",
            "whenever",
            "wherever",
            "whoever",
            "widely",
        }

        # Initialize spaCy model for better semantic similarity
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            self.nlp = None

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            strip_accents="unicode",
        )

        # Update weights to include new metrics
        self.keyword_weights = {
            "keyword_coverage": 0.2,
            "content_similarity": 0.2,
            "keyword_similarity": 0.15,
            "semantic_similarity": 0.15,
            "sentence_structure": 0.1,
            "readability_score": 0.1,
            "filler_word_penalty": 0.1,
        }

        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()
            logger.warning("Failed to load stopwords, using empty set")

    def calculate_scores(
        self, master_transcript: str, candidate_transcript: str
    ) -> Dict[str, float]:
        """Calculate all similarity scores between master and candidate transcripts"""
        try:
            if not master_transcript or not candidate_transcript:
                raise ValueError("Empty transcripts are not allowed")

            if (
                len(master_transcript.split()) < 3
                or len(candidate_transcript.split()) < 3
            ):
                raise ValueError("Transcripts too short (minimum 3 words required)")

            # Clean and normalize transcripts
            master_clean = self._clean_text(master_transcript)
            candidate_clean = self._clean_text(candidate_transcript)

            # Calculate metrics with validation
            scores = {
                "keyword_coverage": self._calculate_keyword_coverage(
                    master_clean, candidate_clean
                ),
                "content_similarity": self._calculate_content_similarity(
                    master_clean, candidate_clean
                ),
                "keyword_similarity": self._calculate_keyword_similarity(
                    master_clean, candidate_clean
                ),
                "semantic_similarity": self._calculate_semantic_similarity(
                    master_clean, candidate_clean
                ),
                "sentence_structure": self._calculate_sentence_structure_similarity(
                    master_clean, candidate_clean
                ),
                "readability_score": self._calculate_readability_score(candidate_clean),
                "filler_word_penalty": self._calculate_filler_penalty(candidate_clean),
            }

            scores["aggregate_score"] = self._aggregate_scores(scores)
            return scores

        except Exception as e:
            logger.error(f"Error in calculate_scores: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text with enhanced preprocessing"""
        # Convert to lowercase
        text = text.lower()

        # Handle contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'ll": " will",
            "'ve": " have",
            "'m": " am",
            "'re": " are",
            "'s": " is",
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        # Remove punctuation but keep sentence boundaries
        text = re.sub(
            r"([.!?])\s*", r"\1 ", text
        )  # Ensure space after sentence endings
        text = re.sub(r"[^\w\s.!?]", " ", text)  # Keep basic punctuation

        # Normalize whitespace
        text = " ".join(text.split())
        return text.strip()

    def _calculate_keyword_coverage(self, master: str, candidate: str) -> float:
        """Calculate keyword coverage using TF-IDF with ngram weighting"""
        try:
            if not master or not candidate:
                logger.warning("Empty input for keyword coverage calculation")
                return 0.0

            # Get important terms with ngram weighting
            master_tfidf = self.vectorizer.fit_transform([master])
            feature_names = np.array(self.vectorizer.get_feature_names_out())

            if len(feature_names) == 0:
                logger.warning("No features found in vectorizer")
                return 0.0

            # Get term importances
            importances = master_tfidf.toarray()[0]
            important_mask = importances > 0
            important_terms = feature_names[important_mask]
            term_weights = importances[important_mask]

            if len(important_terms) == 0:
                logger.warning("No important terms found in master text")
                return 0.0

            # Weight terms by their importance
            candidate_terms = set(self._clean_text(candidate).split())
            weighted_coverage = sum(
                weight
                for term, weight in zip(important_terms, term_weights)
                if term in candidate_terms
            )
            total_weight = sum(term_weights)

            coverage = weighted_coverage / total_weight if total_weight > 0 else 0.0
            return max(0.0, min(1.0, coverage))

        except Exception as e:
            logger.error(f"Error calculating keyword coverage: {str(e)}")
            return 0.0

    def _calculate_content_similarity(self, master: str, candidate: str) -> float:
        """Calculate content similarity using TF-IDF and cosine similarity"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([master, candidate])
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            return 0.0

    def _calculate_keyword_similarity(self, master: str, candidate: str) -> float:
        """Calculate semantic similarity between keywords using WordNet"""
        try:
            # Extract keywords (nouns and verbs) from both texts
            def get_keywords(text: str) -> Set[str]:
                words = nltk.word_tokenize(text.lower())
                pos_tags = nltk.pos_tag(words)
                return {
                    word
                    for word, pos in pos_tags
                    if pos.startswith(("NN", "VB")) and word not in self.stop_words
                }

            master_keywords = get_keywords(master)
            candidate_keywords = get_keywords(candidate)

            if not master_keywords or not candidate_keywords:
                return 0.0

            # Calculate semantic similarity using WordNet
            total_similarity = 0
            count = 0

            for master_word in master_keywords:
                master_synsets = wordnet.synsets(master_word)
                if not master_synsets:
                    continue

                for candidate_word in candidate_keywords:
                    candidate_synsets = wordnet.synsets(candidate_word)
                    if not candidate_synsets:
                        continue

                    # Get maximum similarity between word pairs
                    similarities = [
                        master_syn.path_similarity(candidate_syn)
                        for master_syn in master_synsets
                        for candidate_syn in candidate_synsets
                        if master_syn.path_similarity(candidate_syn) is not None
                    ]

                    if similarities:
                        total_similarity += max(similarities)
                        count += 1

            return total_similarity / count if count > 0 else 0.0

        except Exception as e:
            logger.error(f"Error in keyword similarity: {str(e)}")
            return 0.0

    def _calculate_sentence_structure_similarity(
        self, master: str, candidate: str
    ) -> float:
        """Calculate similarity in sentence structure using dependency parsing"""
        try:
            if self.nlp is None:
                return 0.0

            # Parse sentences
            master_doc = self.nlp(master)
            candidate_doc = self.nlp(candidate)

            # Get dependency patterns for each sentence
            def get_dep_pattern(sent):
                return [(token.dep_, token.pos_) for token in sent]

            master_patterns = [get_dep_pattern(sent) for sent in master_doc.sents]
            candidate_patterns = [get_dep_pattern(sent) for sent in candidate_doc.sents]

            # Calculate pattern similarity
            total_similarity = 0
            comparisons = 0

            for master_pat in master_patterns:
                for candidate_pat in candidate_patterns:
                    # Compare patterns using Levenshtein distance
                    similarity = 1 - (
                        self._levenshtein_distance(str(master_pat), str(candidate_pat))
                        / max(len(str(master_pat)), len(str(candidate_pat)))
                    )
                    total_similarity += similarity
                    comparisons += 1

            return total_similarity / comparisons if comparisons > 0 else 0.0

        except Exception as e:
            logger.error(f"Error in sentence structure similarity: {str(e)}")
            return 0.0

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score using TextBlob"""
        try:
            # Calculate average sentence length
            sentences = sent_tokenize(text)
            words = text.split()
            avg_sentence_length = len(words) / len(sentences) if sentences else 0

            # Calculate lexical diversity
            unique_words = len(set(words))
            total_words = len(words)
            lexical_diversity = unique_words / total_words if total_words > 0 else 0

            # Use TextBlob for polarity and subjectivity
            blob = TextBlob(text)
            sentiment_score = (1 + blob.sentiment.polarity) / 2  # Normalize to 0-1

            # Combine metrics into readability score
            readability_score = (
                0.4
                * (1 - min(avg_sentence_length / 20, 1))  # Shorter sentences are better
                + 0.4 * lexical_diversity  # Higher lexical diversity is better
                + 0.2 * sentiment_score  # More neutral sentiment is better
            )

            return max(0.0, min(1.0, readability_score))

        except Exception as e:
            logger.error(f"Error in readability score: {str(e)}")
            return 0.0

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _calculate_filler_penalty(self, text: str) -> float:
        """Calculate filler word penalty"""
        try:
            words = text.lower().split()
            if not words:
                return 0.0

            filler_count = sum(1 for word in words if word in self.filler_words)
            # Calculate penalty (1 = no penalty, 0 = maximum penalty)
            penalty = max(0.0, 1.0 - (filler_count / len(words)) * 2)
            return penalty
        except Exception:
            return 0.0

    def _calculate_semantic_similarity(self, master: str, candidate: str) -> float:
        """Calculate semantic similarity using spaCy's word embeddings"""
        try:
            if self.nlp is None:
                return self._calculate_content_similarity(master, candidate)

            # Use spaCy's document similarity
            doc1 = self.nlp(master)
            doc2 = self.nlp(candidate)
            return float(doc1.similarity(doc2))
        except Exception as e:
            logger.error(f"Error in semantic similarity: {str(e)}")
            return self._calculate_content_similarity(master, candidate)

    def _get_ngrams(
        self, text: str, n_range: range = range(1, 4)
    ) -> Dict[int, Set[str]]:
        """Get n-grams from text"""
        words = text.split()
        ngrams = {}
        for n in n_range:
            ngrams[n] = set(
                " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
            )
        return ngrams

    def _aggregate_scores(self, scores: Dict[str, float]) -> float:
        """Calculate weighted average of scores"""
        try:
            # Calculate weighted sum excluding aggregate_score
            weighted_sum = sum(
                scores[metric] * weight
                for metric, weight in self.keyword_weights.items()
                if metric in scores
            )
            total_weight = sum(
                weight
                for metric, weight in self.keyword_weights.items()
                if metric in scores
            )
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        except Exception:
            return 0.0
