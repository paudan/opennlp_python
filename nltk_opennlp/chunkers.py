# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the OpenNLP chunker
#
# Copyright (C) Paulius Danenas
# Author: Paulius Danenas <danpaulius@gmail.com>

"""
A Python module for interfacing with the Apache OpenNLP package and obtaining chunk parse tree
"""

import gc
import os
import sys
import re
import string
from subprocess import Popen, PIPE
from nltk.internals import find_binary
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree


class OpenNLPChunker(ChunkParserI):

    use_punc_tag = False

    def __init__(self, path_to_bin=None, path_to_chunker=None, verbose=False, use_punc_tag=False):
        """
        Initialize the OpenNLPTagger.
        :param path_to_bin: Path to bin directory of OpenNLP installation
        :param path_to_chunker: The path to OpenNLP POS tagger .bin file.
        :param use_punc_tag: Whether standalone punctuation marks should be tagged using PUNC tag

        """
        if path_to_chunker is None:
            raise LookupError('OpenNLP model file is not set!')
        self._model_path = path_to_chunker
        self.use_punc_tag = use_punc_tag

        opennlp_paths = ['.', '/usr/bin', '/usr/local/apache-opennlp', '/opt/local/apache-opennlp', '~/apache-opennlp']
        opennlp_paths = list(map(os.path.expanduser, opennlp_paths))

        opennlp_bin_name = "opennlp"
        if sys.platform.startswith("win"):
            opennlp_bin_name += ".bat"
        try:
            self._opennlp_bin = find_binary(opennlp_bin_name, os.path.join(path_to_bin, opennlp_bin_name),
                                          env_vars=('OPENNLP_HOME', 'OPENNLP'),
                                          searchpath=opennlp_paths, verbose=verbose)
        except LookupError:
            print('Unable to find the Apache OpenNLP run file!')


    def __parse_punc_tags__(self, output):
        if self.use_punc_tag == True:
            return re.sub(r'\(\s*([' + string.punctuation + ']+)\s*/\s*([' + string.punctuation + ']+)\s*\)',
                          r'(\1/PUNC)', output)
        return output


    def parse(self, tokens):
        _input = ' '.join([token[0] + "_" + token[1] for token in tokens])

        gc.collect()
        p = Popen([self._opennlp_bin, "ChunkerME", self._model_path],
                  shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)

        if sys.version_info >= (3,):
            (stdout, stderr) = p.communicate(bytes(_input, 'UTF-8'))
            stdout = stdout.decode('utf-8')
        else:
            (stdout, stderr) = p.communicate(_input)

        # Check the return code.
        if p.returncode != 0:
            raise OSError('OpenNLP command failed!')

        # Clean the execution time information
        output = re.sub(r"\nExecution time:(.*)$", "", stdout)
        # Transform into compatible parse tree string
        output = output.replace("[", "(").replace("]", " )")
        pattern = re.compile(r'\s+([^_\(\)]+)_([^_\(\)]+)\s+')
        output = re.sub(pattern, r' (\1/\2) ', output)
        output = re.sub(pattern, r' (\1/\2) ', output)
        # Add punctuation tags if use_punc_tag is set
        output = self.__parse_punc_tags__(output)
        output = "(S {} )".format(output)
        try:
            parse = Tree.fromstring(output)
        except Exception:
            parse = None
        return parse


class OpenNERChunker(OpenNLPChunker):

    def __init__(self, path_to_ner_model = None, *args, **kwargs):
        OpenNLPChunker.__init__(self, *args, **kwargs)
        self._ner_model = path_to_ner_model


    def parse(self, tokens):

        treeObj = OpenNLPChunker.parse(self, tokens)
        treeStr = treeObj.__str__()

        _input = ' '.join([token[0] for token in tokens])

        gc.collect()
        p = Popen([self._opennlp_bin, "TokenNameFinder", self._ner_model],
                  shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)

        if sys.version_info >= (3,):
            (stdout, stderr) = p.communicate(bytes(_input, 'UTF-8'))
            stdout = stdout.decode('utf-8')
        else:
            (stdout, stderr) = p.communicate(_input)

        # Check the return code.
        if p.returncode != 0:
            raise OSError('OpenNLP command failed!')

        # Clean the execution time information
        output = re.sub(r"\nExecution time:(.*)$", "", stdout)
        # Extract entities
        tag_match = re.compile('<START:(.*?)>(.*?)<END>')
        matches = tag_match.findall(output)
        for match in matches:
            tagname = match[0].upper()
            pattern = '\s+'.join('\(\s*'+ token + '\s*/\s*[A-Z]+\s*\)'
                                 for token in match[1].strip().split(' '))
            tpattern = '(?P<token>'+ pattern + ')'
            tagged_pattern = '('+ tagname + ' (\g<token>))'
            treeStr = re.sub(tpattern, tagged_pattern, treeStr, flags=re.UNICODE)
            # "Move up" NER tags when possible
            treeStr = re.sub('\(\s*NP\s+(?P<subtree>\(' + tagname + '(.*)\s*\)\s*\))\s*\)',
                             '\g<1>', treeStr, flags=re.UNICODE)
        # Add punctuation tags if use_punc_tag is set
        treeStr = self.__parse_punc_tags__(treeStr)
        try:
            parse = Tree.fromstring(treeStr)
        except Exception:
            parse = None
        return parse


class OpenNERChunkerMulti(OpenNLPChunker):

    def __init__(self, ner_models = [], *args, **kwargs):
        OpenNLPChunker.__init__(self, *args, **kwargs)
        self._ner_models = ner_models


    def parse(self, tokens):

        treeObj = OpenNLPChunker.parse(self, tokens)
        treeStr = treeObj.__str__()

        _input = ' '.join([token[0] for token in tokens])

        for model in self._ner_models:
            gc.collect()
            p = Popen([self._opennlp_bin, "TokenNameFinder", model],
                      shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)

            if sys.version_info >= (3,):
                (stdout, stderr) = p.communicate(bytes(_input, 'UTF-8'))
                stdout = stdout.decode('utf-8')
            else:
                (stdout, stderr) = p.communicate(_input)

            # Check the return code.
            if p.returncode != 0:
                raise OSError('OpenNLP command failed!')

            # Clean the execution time information
            output = re.sub(r"\nExecution time:(.*)$", "", stdout)
            # Extract entities
            tag_match = re.compile('<START:(.*?)>(.*?)<END>')
            matches = tag_match.findall(output)
            for match in matches:
                tagname = match[0].upper()
                pattern = '\s+'.join('\(\s*'+ token + '\s*/\s*[A-Z]+\s*\)'
                                     for token in match[1].strip().split(' '))
                tpattern = '(?P<token>'+ pattern + ')'
                tagged_pattern = '('+ tagname + ' (\g<token>))'
                treeStr = re.sub(tpattern, tagged_pattern, treeStr, flags=re.UNICODE)
                # "Move up" NER tags when possible
                treeStr = re.sub('\(\s*NP\s+(?P<subtree>\(' + tagname + '(.*)\s*\)\s*\))\s*\)',
                                 '\g<1>', treeStr, flags=re.UNICODE)
        # Add punctuation tags if use_punc_tag is set
        treeStr = self.__parse_punc_tags__(treeStr)
        try:
            parse = Tree.fromstring(treeStr)
        except Exception:
            parse = None
        return parse
