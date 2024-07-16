from typing import List, Tuple, Dict, Any
from keybert import KeyBERT
from config import BERT_MODEL

def extract_keywords_from_doc(
    doc: str,
    model_name: str = BERT_MODEL,
    **kwargs: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """
    ## Summary
    Extract keywords from a document using the KeyBERT model.

    ## Parameters:
        doc (str): The document from which to extract keywords.
        model_name (str): The name of the model to use. Default is "paraphrase-multilingual-MiniLM-L12-v2".
        **kwargs (Dict[str, Any]): Additional keyword arguments for the extract_keywords method.
            Possible keyword arguments include:
                - top_n (int): The number of top keywords to return.
                - keyphrase_ngram_range (Tuple[int, int]): The ngram range for the keyphrases.
                - stop_words (str): The stop words to use.
                - use_maxsum (bool): Whether to use Max Sum Similarity.
                - use_mmr (bool): Whether to use Maximal Marginal Relevance.
                - diversity (float): The diversity parameter for MMR.
                - nr_candidates (int): The number of candidates for Max Sum Similarity.

    ## Returns:
        List[Tuple[str, float]]: A list of tuples containing keywords and their corresponding scores.

    ## Example:
        doc = \"\"\"
        Supervised learning is the machine learning task of learning a function that
        maps an input to an output based on example input-output pairs. It infers a
        function from labeled training data consisting of a set of training examples.
        In supervised learning, each example is a pair consisting of an input object
        (typically a vector) and a desired output value (also called the supervisory signal).
        A supervised learning algorithm analyzes the training data and produces an inferred function,
        which can be used for mapping new examples. An optimal scenario will allow for the
        algorithm to correctly determine the class labels for unseen instances. This requires
        the learning algorithm to generalize from the training data to unseen situations in a
        'reasonable' way (see inductive bias).
        \"\"\"

        keywords = extract_keywords_from_doc(
            doc,
            top_n=10,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            use_maxsum=True,
            nr_candidates=20
        )
        print(keywords)
    """
    kw_model = KeyBERT(model=model_name)
    keywords = kw_model.extract_keywords(doc, **kwargs)
    return keywords

if __name__ == "__main__":

    # Example usage
    doc = """
    Supervised learning is the machine learning task of learning a function that
    maps an input to an output based on example input-output pairs. It infers a
    function from labeled training data consisting of a set of training examples.
    In supervised learning, each example is a pair consisting of an input object
    (typically a vector) and a desired output value (also called the supervisory signal).
    A supervised learning algorithm analyzes the training data and produces an inferred function,
    which can be used for mapping new examples. An optimal scenario will allow for the
    algorithm to correctly determine the class labels for unseen instances. This requires
    the learning algorithm to generalize from the training data to unseen situations in a
    'reasonable' way (see inductive bias).
    """

    # Example of passing additional keyword arguments
    keywords = extract_keywords_from_doc(
        doc,
        top_n=10,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        use_maxsum=True,
        nr_candidates=20
    )
    print(keywords)
