   # Can the NLP be Used in Crime Fighting #

Project 3                                          Bootcamp Final Project



### Contents

1.Introduction

2.A Preliminary look at the WP literature.ipynb

    a.  Frequency Distribution

3.2. Listing the words out of the manifesto text.ipynb

4.3. Bag of words for better insight. ipynb

    b. Noise and Stop Words

    c. Tokenization

    d. Normalization

    e. Bag of Words & ngrams

5.4. Unabomber\_Manifesto.ipynb

    f. Vectorization

    g. TF-IDF

6. Conclusion

7. References

_______________________________________________________________________________
### Introduction: ###

Ladies and gentlemen, in other words, johnny and my dear colleagues,
Welcome to an evening with Spacy, Displacy and the crazy words token,
with their colleagues like NTLK, in the occasion of Natural Language
Processing. Well, that is not our whole topic tonight, anyway. It is
much wider & deeper.

The language spoken at a crime site by criminal is a thumb print left
behind by the criminal, though it is not as conclusive as a fingerprint
or even like DNA codes. The words and the way the NL used by the
criminal were of great interest for investigators, always. One of the
famous cases cracked with the Natural Language used by the criminal to
convey his communication is “Unabomber’s” case. From 1971 to 1998, this
highly educated Berkeley University professor was a sensational topic
for the media and a concern with anxiety for American people. He has
been sending bombs on postal packages, to the people who didn't know him
before that. Later, he released a 300-page manifesto to the Washington
Post and New York Times. Washington post released a supplement of it
with the collaboration of NYT, because AG Janet Reno and FBI consulted
them to do so. The FBI used these writings to figure out some clue of
who the culprit would be. Based on the FBI’s descriptions & sketch,
David Kaczynski recognized that the criminal was his brother, Ted
Kaczynski, PhD. and he helped FBI to apprehend Ted.

The FBI used mainly its Internal and external Language Pundits to decode
the secret IDs hidden in those texts, in those days. That time most of
the Natural Language Processing technology we learned here was not
available.

Voila; so “If NLP could be used in crime fighting” is the topic,
tonight, for our project. We will download a copy of the Unabomber’s
manifesto and analyses with some of the NLP techniques we learned in
this class to see if we can make any prediction with reasonable
accuracy.

Natural Language is the Language used by Humans to communicate within
them. NLP is about feeding the information that is present in
unstructured textual writing or spoken form to a computer with the
intention of having it mechanically analyzed. NLP is one side of the
coin of Artificial Intelligence. The other one is machine learning.
Artificial Intelligence, which is an attempt to make the machine to
think & act on its own, like humans, instead of just to repeat or
copycat or even be programmed by human beings. Therefore, the machine
learning has become the other part of AI. Our project is about using the
data available out in the open communication channels and using it to
fight crime. Sources to this data is Printed Press, on-line media,
social media and other channels. NLP can be used to character
recognition, sentiment analysis, & dot products. Our concentration here
is about character recognition.

Probably NLP is the most widely used Fintech section we came through. On
the other hand, it is said to be the most difficult technology. We call
this process text mining. In text mining we start with Information
extraction, documents classification & clustering, information retrieval
and finally NLP. We will be using many tools one after another to study
this case in detail and keep comparing the outputs.

So, let’s start with our basic trial. Here we are going to download Ted
Kaczynski’s manifesto from the Washington Post and convert it from html
to a processable text.

### 1. A Preliminary look at the WP literature.ipynb (look at the notebook) ###

Once we downloaded, we used the BeautifulSoup tool to remove the html
content and save it as a text file, locally. Local copy is convenient
for trials instead of repeatedly downloading from WP. In this notebook
we tried to assess the nature of the file. It has 4808 words and the
highest frequency word “the” alone has appeared around 1735 times. It is
a massive document of 300 pages with WP’s introduction, additional to
that. We used NTLK’s FreqDist to analyze the document. Then we dropped
the words below two letters. At that point, the words “that, have,
society…” have become the highest frequency words. We had successfully
eliminated punctuation characters by selecting words above one letter,
but still the stop words are marooning the regular words.

a)- Frequency Distribution:

A frequency distribution can be defined as a function, mapping from each
sample to the number of times that sample occurred as an outcome.
Frequency distributions are generally constructed by running several
experiments and incrementing the count for a sample every time it is an
outcome of an experiment. Thus, having understood what the frequency is
in this notebook, we can move to other tools in other notebooks.

### 2. Listing the words out of the manifesto text.ipynb

Now we will go for some more tools which will help us to identify the
stop words and drop them out. We downloaded the stop words from NLTK and
used a loop to eliminate them. The notebook is comparing the results. We
introduced the word cloud too. It did a better job than work we did
earlier. Look at the No. 2 Notebook.

### 3. Bag of words for better insight. ipynb

Here we dropped out all manual loops that we have been using. So, to do
the cleaning work, we are using the built-in tools. We are using the
market leaders like NTLK for word\_tokenizer, WordNetLemmatizer, Then we
added NTLK “stop words” and spacy.lang.en.stop\_words together to get a
larger list. This really got rid of the “would” from the plot and “will”
from the Word Cloud. We used RegEx to clean it.

Let's look at the some of theory behind this Notebook 3. in a few lines.

b)- Noise and Stop Words:

In regular sentences Noisy data can be defined as file header, footer,
HTML, XML, markup data….. As these types of data are not meaningful and
do not provide any information so it is a necessity to remove these
types of noisy data. In python HTML, XML can be removed by BeautifulSoup
library while markup, header can be removed by using regular expression
(RegEx)

c)- Tokenization:

In tokenization we convert a group of sentences into a token. It is also
called text segmentation or lexical analysis. It is basically splitting
data into small chunks of words. For example- We have a sentence —
“well, you can’t eat your cake and have it too!”. After tokenization
this sentence will become -\[‘well’, ‘,’, ‘you’, ‘can’t’ , ‘eat’,
‘your’, ‘cake’, ‘and’, ‘have’, ‘it’, ‘too’ ,’!’\]. Sentence
tokenization, word tokenization, regex ionization and blank line
tokenization.

d)- Normalization:

Before going to normalization, let us first closely observe the output
of tokenization. Will tokenization output be considered as final output?
Can we extract more meaningful information from tokenized data?

In tokenization we came across various words such as punctuation, stop
words, upper case words and lower-case words. After tokenization we are
not focused on text level, but on word level. Further, by doing
lemmatization we can convert the tokenized words to more meaningful
words’ stems. Please look at this example: “I said to go. She feared of
going alone. His brother went that way. Then she too was gone. Then I
said to myself, see she cannot go alone.” Here I use the word “go” five
times, “to go”, “going”, “went”, “gone”, and as last “go”. The
lemmatization helps to simplify the list of words by converting them to
their stem format.

e)- Bag of Words & ngrams

Bag of word is a basic model used in natural language processing. Like
the merchandise in a bag, at first, it has no unique arrangement. In
unigrams, they are just one cluster of words. But bigram and other grams
try to recognize their order in the sentence. The word “have” in unigram
can be “‘have’, ‘not’” in bigram. Then, there is the idea you form from
the NLP analysis of the document is reversed.

### 4.Unabomber\_Manifesto.ipynb ###

f)- Vectorization:

Word Embeddings or Word Vectorization is a methodology in NLP to map
words or phrases from vocabulary to a corresponding vector of real
numbers which is used to find word predictions, word
similarities/semantics. An integer-based array form is used to show the
position & status of the word in the sentence (or even in doc). I didn’t
use it here because Unabomber text is one single doc of 4808 words long
and creating an array for it in my computer is out of question. But we
will use it in the notebook-4 in formulas.

g)- TF-IDF:

The last technology we are using to analyses the manifesto is TF-IDF.

TF-IDF stands for “Term Frequency — Inverse Data Frequency”.

Term Frequency (tf): It is the ratio of the number of times the word
appears in a document compared to the total number of words in that
document. It increases as the number of occurrences of that word within
the document increases. Please note, it is important that each document
has its own tf.

TF: This is the ratio of a word’s occurrence in a document.

<img src="media\image1.png" style="width:3.57292in;height:1.26042in" />

IDF: This is the ratio of inversed occurrence of the word’s occurrence
in the corpus compared to the number of documents in the corpus. Because
this the inversed ratio, for rare words, the number can be very big. To
manage the size of the result it is converted into the logarithmic
number. The log will account the result in “to the power to ten”,
usually a small number represent the large million or billion numbers.

<img src="media\image2.png" style="width:4.06042in;height:1.27986in" alt="Text, letter Description automatically generated" />

TF-IDF is the one obtained combing both TF and IDF

<img src="media\image3.png" style="width:5.25694in;height:1.33056in" alt="Text Description automatically generated" />

Legend:

<img src="media\image4.png" style="width:3.51042in;height:0.89792in" alt="Text Description automatically generated" />

Look at the notebook 4. Unabomber\_Manifesto.ipynb

### Conclusion: ###

We were able to see Unabomber’s manifestos were written on good topics.
But when he was psychologically analyzed, it was found that he had
remained not having treated his illness. This made him hate human
civilization and he wanted to return to natural forest dwelling. He used
violence to force the American community for this end.

As we said earlier NLP is now everywhere we move. Very basic is, if you
up a word’s meaning or audio sound, it is facilitated by NLP. Every one
of your searches in google is responded to by NLP. Most of the telephone
replies or chatbot conversions are provided by NLP. It is in medical,
engineering, law ….crime investigation…. Almost everywhere. The
Unabomber case was cracked with the help of understanding the Natural
Language. It was done manually reading the text that. Now NLP has grown
a lot. We used many NLP tools. Almost all of them gave pretty same
answer. But they are still further away from giving the same manual
answer that FBI obtained that time. For example, an agent who noted the
term he had used for payment grass was used by a specific Chicago
district, in all America. This is how his location was found out. His
murders at Berkeley university revealed his connection to that
university. In all our analysis, it was pretty clear of his thinking,
but we were not able to have any clue on the above matter. We know
thieves use gloves to hide their identity. NLP has to be developed to
tear off those masks to provide better help.

### References: ### 

1\.
https://www.josharcher.uk/blog/industrial-society-and-its-future-an-analysis/

2\.
https://codereview.stackexchange.com/questions/181496/analysis-of-the-most-common-and-salient-words-in-a-text

3\.
https://betterprogramming.pub/a-friendly-guide-to-nlp-tf-idf-with-python-example-5fcb26286a33

4\.
https://medium.com/@krishvictor/fighting-crime-with-text-analytics-2bcbaf7ff6c4

5\.
https://www.washingtonpost.com/wp-srv/national/longterm/unabomber/manifesto.text.htm

6\. https://intellipaat.com/blog/what-is-natural-language-processing/

7\. Wikipedia.com

8\. many lines were adopted from turorials of Triology

9\. https://www.datacamp.com/community/tutorials/

10\. https://www.geeksforgeeks.org/python-projects-beginner-to-advanced/

11\.
<https://stackoverflow.com/questions/3217222/beginner-python-practice>

12\. And others (May not be the complete list)

13\.
https://medium.com/free-code-camp/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3
