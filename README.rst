nltk-opennlp
============

NLTK interface with Apache OpenNLP

Copyright (C) 2017 Paulius Danenas

Dependencies
------------

-  `Apache OpenNLP <https://opennlp.apache.org/>`__
-  Python 2.7 or Python 3
-  `NLTK <http://nltk.org/>`__

Tested with OpenNLP 1.8 (using models built with 1.5), Python 2.7/3.5 and NLTK 3.2.4

Installation
------------

Before you install the ``nltk-opennlp`` package please ensure you
have downloaded and installed the `Apache OpenNLP <https://opennlp.apache.org/>`__
itself. You will also need different tagger/chunker models; some of them are provided in
`this repository <http://opennlp.sourceforge.net/models-1.5/>`__

Usage
-----

Tagging a sentence from Python:

.. code:: python

    from nltk_opennlp.taggers import OpenNLPTagger

    tt = OpenNLPTagger(language='en',
                        path_to_bin=os.path.join('/path/to/opennlp/installation', 'bin'),
                        path_to_model=os.path.join('/path/to/opennlp/models', 'en-pos-maxent.bin'))
    phrase = 'Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .'
    sentence = tt.tag(phrase)

The output is a list of (token, tag):

::

    [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'),
    ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'),
    ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'),
    ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'),
    ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')]


Chunking the same sentence from Python will produce a parse tree:

.. code:: python

    from nltk_opennlp.chunkers import OpenNLPChunker
    from nltk_opennlp.taggers import OpenNLPTagger

    tt = OpenNLPTagger(language='en',
                       path_to_bin=os.path.join('/path/to/opennlp/installation', 'bin'),
                       path_to_model=os.path.join('/path/to/opennlp/models', 'en-pos-maxent.bin'))
    phrase = 'Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .'
    sentence = tt.tag(phrase)
    cp = OpenNLPChunker(path_to_bin=os.path.join('/path/to/opennlp/installation', 'bin'),
                        path_to_chunker=os.path.join('/path/to/opennlp/models', 'en-chunker.bin'))
    print(cp.parse(sentence))

The output is a parse tree:

::

    (S
      (NP Pierre/NNP Vinken/NNP)
      ,/,
      (NP 61/CD years/NNS)
      (ADJP old/JJ)
      ,/,
      (VP will/MD join/VB)
      (NP the/DT board/NN)
      (PP as/IN)
      (NP a/DT nonexecutive/JJ director/NN)
      (NP Nov./NNP 29/CD)
      ./.)

Note, that is possible to use PUNC tag to tag standalone punctuation marks, using ``use_punc_tag`` parameter. After setting this param, the output would be come as following:

::

    (S
      (PERSON Pierre/NNP Vinken/NNP)
      ,/PUNC
      (NP 61/CD years/NNS)
      (ADJP old/JJ)
      ,/PUNC
      (VP will/MD join/VB)
      (PERSON Martin/NNP Vinken/NNP)
      (PP as/IN)
      (NP a/DT nonexecutive/JJ director/NN)
      (NP Nov./NNP 29/CD)
      ./PUNC)

Tagging a german sentence from Python is similar, just need to use diferent language and pre-trained model:

.. code:: python

    from nltk_opennlp.taggers import OpenNLPTagger

    tt = OpenNLPTagger(language='de',
                        path_to_bin=os.path.join('/path/to/opennlp/installation', 'bin'),
                        path_to_model=os.path.join('/path/to/opennlp/models', 'de-pos-maxent.bin'))
    tt.tag('Das Haus hat einen großen hübschen Garten.')

The output is a list of (token, tag):

::

    [('Das', 'ART'), ('Haus', 'NN'), ('hat', 'VAFIN'), ('einen', 'ART'), (
    'großen', 'ADJA'), ('hübcbschen', 'ADJA'), ('Garten.', 'NN')]

Named entity recognition (NER)
------------------------------

This module also supports named entity recognition, which allows to tag particular types of entities. Again, chunking
is performed on the set of (token, tag) entries (note, that NLTK taggers could be used instead of ``OpenNLPTagger``):

.. code:: python

    from nltk_opennlp.chunkers import OpenNERChunker

    language='en'
    tt = OpenNLPTagger(language=language,
                       path_to_bin=os.path.join(opennlp_dir, 'apache-opennlp', 'bin'),
                       path_to_model=os.path.join(opennlp_dir, 'opennlp_models', 'en-pos-maxent.bin'))
    phrase = 'Pierre Vinken , 61 years old , will join Martin Vinken as a nonexecutive director Nov. 29 .'
    sentence = tt.tag(phrase)
    cp = OpenNERChunker(path_to_bin=os.path.join(opennlp_dir, 'apache-opennlp', 'bin'),
                        path_to_chunker=os.path.join(opennlp_dir, 'opennlp_models', '{}-chunker.bin'.format(language)),
                        path_to_ner_model=os.path.join(opennlp_dir, 'opennlp_models', '{}-ner-person.bin'.format(language)),
                        use_punc_tag=True)
    print(cp.parse(sentence))

The output is a chunk parse tree with particular types of entities:

::

    (S
      (PERSON Pierre/NNP Vinken/NNP)
      ,/,
      (NP 61/CD years/NNS)
      (ADJP old/JJ)
      ,/,
      (VP will/MD join/VB)
      (PERSON Martin/NNP Vinken/NNP)
      (PP as/IN)
      (NP a/DT nonexecutive/JJ director/NN)
      (NP Nov./NNP 29/CD)
      ./.)

A multi-tagger option is similar, except that it allows to set multiple NER models for tagging:

.. code:: python

    from nltk_opennlp.chunkers import OpenNERChunkerMulti

    language='en'
    phrase = 'John Haddock , 32 years old male , travelled to Cambridge , USA in October 20 while paying 6.50 dollars for the ticket'
    sentence = tt.tag(phrase)
    cp = OpenNERChunkerMulti(language=language,
                        path_to_bin=os.path.join(opennlp_dir, 'apache-opennlp', 'bin'),
                        path_to_chunker=os.path.join(opennlp_dir, 'opennlp_models', '{}-chunker.bin'.format(language)),
                        ner_models=[os.path.join(opennlp_dir, 'opennlp_models', '{}-ner-person.bin'.format(language)),
                                    os.path.join(opennlp_dir, 'opennlp_models', '{}-ner-date.bin'.format(language)),
                                    os.path.join(opennlp_dir, 'opennlp_models', '{}-ner-location.bin'.format(language)),
                                    os.path.join(opennlp_dir, 'opennlp_models', '{}-ner-time.bin'.format(language)),
                                    os.path.join(opennlp_dir, 'opennlp_models', '{}-ner-money.bin'.format(language))])
    print(cp.parse(sentence))

The resuting chunk tree contains multiple types of identified entities:

::

    (S
      (PERSON John/NNP Haddock/NNP)
      ,/,
      (NP 32/CD years/NNS)
      (NP old/JJ male/NN)
      ,/,
      (VP travelled/VBN)
      (PP to/TO)
      (LOCATION Cambridge/NNP)
      ,/,
      (NP USA/NNP)
      (PP in/IN)
      (DATE October/NNP 20/CD)
      (PP while/IN)
      (VP paying/VBG)
      (NP 6.50/CD dollars/NNS)
      (PP for/IN)
      (NP the/DT ticket/NN))