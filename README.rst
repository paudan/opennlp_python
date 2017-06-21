nltk-opennlp
=================

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

    from opennlp_tagger import OpenNLPTagger
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

    tt = OpenNLPTagger(language='en',
                       path_to_bin=os.path.join('/path/to/opennlp/installation', 'bin'),
                       path_to_model=os.path.join('/path/to/opennlp/models', 'en-pos-maxent.bin'))
    phrase = 'Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .'
    sentence = tt.tag(phrase)
    cp = OpenNLPChunker(language='en',
                        path_to_bin=os.path.join('/path/to/opennlp/installation', 'bin'),
                        path_to_model=os.path.join('/path/to/opennlp/models', 'en-chunker.bin'))
    print(cp.parse(sentence))

The output is a parse tree:

::

    (ROOT
      (NP (Pierre NNP) (Vinken NNP))
      (, ,)
      (NP (61 CD) (years NNS))
      (ADJP (old JJ))
      (, ,)
      (VP (will MD) (join VB))
      (NP (the DT) (board NN))
      (PP (as IN))
      (NP (a DT) (nonexecutive JJ) (director NN))
      (NP (Nov. NNP) (29 CD))
      (. .))


Tagging a german sentence from Python is similar, just need to use diferent language and pre-trained model:

.. code:: python

    from opennlp_tagger import OpenNLPTagger
    tt = OpenNLPTagger(language='de',
                        path_to_bin=os.path.join('/path/to/opennlp/installation', 'bin'),
                        path_to_model=os.path.join('/path/to/opennlp/models', 'de-pos-maxent.bin'))
    tt.tag('Das Haus hat einen großen hübschen Garten.')

The output is a list of (token, tag):

::

    [('Das', 'ART'), ('Haus', 'NN'), ('hat', 'VAFIN'), ('einen', 'ART'), (
    'großen', 'ADJA'), ('hübcbschen', 'ADJA'), ('Garten.', 'NN')]
