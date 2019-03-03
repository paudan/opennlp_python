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
from nltk.tree import Tree, ParentedTree


class OpenNLPChunker(ChunkParserI):

    use_punc_tag = False

    def __init__(self, path_to_bin=None, path_to_chunker=None, verbose=False, use_punc_tag=False):
        """
        Initialize the OpenNLPChunker.
        :param path_to_bin: Path to bin directory of OpenNLP installation
        :param path_to_chunker: The path to OpenNLP POS chunker .bin file.
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
            return re.sub(r'\(\s*([' + string.punctuation + ']+)\s+([' + string.punctuation + ']+)\s*\)',
                          r'(PUNC \1)', output)
        return output


    def __perform_parsing__(self, tokens):
        _input = ' '.join([token[0] + "_" + token[1] for token in tokens])

        gc.collect()
        p = Popen([self._opennlp_bin, "ChunkerME", self._model_path],
                  shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)

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
        output = self.__encode__(output)
        output = output.replace("[", "(").replace("]", " )")
        pattern = re.compile(r'\s+([^_\(\)]+)_([^_\(\)]+)\s+')
        output = re.sub(pattern, r' (\2 \1) ', output)
        output = re.sub(pattern, r' (\2 \1) ', output)
        # Add punctuation tags if use_punc_tag is set
        output = self.__parse_punc_tags__(output)
        output = "(S {} )".format(output)
        try:
            parse = Tree.fromstring(output, remove_empty_top_bracketing=True)
        except Exception:
            parse = None
        finally:
            return parse


    def parse(self, tokens):
        parse = self.__perform_parsing__(tokens)
        return self.__get_nltk_parse_tree__(parse)


    __encodings__ = {
        "(": "xleftbrackx", ")": "xrightbrackx"
    }

    def __encode__(self, token):
        if token is None:
            return token
        for enc in self.__encodings__:
            token = token.replace(enc, self.__encodings__[enc])
        return token


    def __decode_(self, token):
        if token is None:
            return token
        inv_map = {v: k for k, v in self.__encodings__.items()}
        for enc in inv_map:
            token = token.replace(enc, inv_map[enc])
        return token


    def __get_nltk_parse_tree__(self, tree):

        def create_tree(tree):
            nodes = []
            for n in tree:
                subtrees = [subtree for subtree in n.subtrees(filter=lambda k: k != n)]
                if len(subtrees) > 0:
                    subnodes = create_tree(n)
                    nodes.append(ParentedTree(n.label(), subnodes))
                else:
                    parent_label = n.parent().label() if n.parent() is not None \
                                                         and n.parent().label() not in ['S', 'ROOT'] else None
                    nodes.append(ParentedTree(parent_label, [(self.__decode_(n[0]), self.__decode_(n.label()))]))
            return nodes

        def move_up(tree):
                
            for i in range(len(tree[:])):
                n = tree[i]
                if isinstance(n, Tree):
                    subtrees = [subtree for subtree in n.subtrees(filter=lambda k: k != n or k.label() is None)]
                    if i == 0:
                        subtrees = subtrees[::-1]
                    for subtree in subtrees:
                        if subtree.label() == n.label() or subtree.label() is None:
                            tmp = subtree
                            parent = subtree.parent()
                            parent.remove(tmp)
                            subsub = [s for s in subtree.subtrees(filter=lambda k: k != subtree)]
                            if len(subsub) == 0:
                                for k in range(len(tmp.leaves())-1, -1, -1):
                                    parent.insert(i, tmp.leaves()[k])
                            else:
                                move_up(n)
            return tree

        tree = ParentedTree.convert(tree)
        new_tree = ParentedTree('S', create_tree(tree))
        return move_up(new_tree)


class OpenNERChunker(OpenNLPChunker):

    def __init__(self, path_to_ner_model = None, *args, **kwargs):
        OpenNLPChunker.__init__(self, *args, **kwargs)
        self._ner_model = path_to_ner_model


    def parse(self, tokens):

        treeObj = self.__perform_parsing__(tokens)
        treeStr = treeObj.__str__()

        _input = ' '.join([token[0] for token in tokens])

        gc.collect()
        p = Popen([self._opennlp_bin, "TokenNameFinder", self._ner_model],
                  shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)

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
            pattern = '\s+'.join('\(\s*[A-Z]+\s+' + token + '\s*\)'
                                 for token in match[1].strip().split(' '))
            tpattern = '(?P<token>'+ pattern + ')'
            tagged_pattern = '('+ tagname + ' \g<token>)'
            treeStr = re.sub(tpattern, tagged_pattern, treeStr, flags=re.UNICODE)
            # "Move up" NER tags when possible
            treeStr = re.sub('\(\s*NP\s+(?P<subtree>\(' + tagname + '(.*)\s*\)\s*\))\s*\)',
                             '\g<1>', treeStr, flags=re.UNICODE)
        # Add punctuation tags if use_punc_tag is set
        treeStr = self.__parse_punc_tags__(treeStr)
        try:
            parse = Tree.fromstring(treeStr, remove_empty_top_bracketing=True)
            parse = self.__get_nltk_parse_tree__(parse)
        except Exception:
            parse = None
        finally:
            return parse



class OpenNERChunkerMulti(OpenNLPChunker):

    def __init__(self, ner_models = [], *args, **kwargs):
        OpenNLPChunker.__init__(self, *args, **kwargs)
        self._ner_models = ner_models


    def parse(self, tokens):

        treeObj = self.__perform_parsing__(tokens)
        treeStr = treeObj.__str__()

        _input = ' '.join([token[0] for token in tokens])

        for model in self._ner_models:
            gc.collect()
            p = Popen([self._opennlp_bin, "TokenNameFinder", model],
                      shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)

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
                pattern = '\s+'.join('\(\s*[A-Z]+\s+' + token + '\s*\)'
                                     for token in match[1].strip().split(' '))
                tpattern = '(?P<token>'+ pattern + ')'
                tagged_pattern = '('+ tagname + ' \g<token>)'
                treeStr = re.sub(tpattern, tagged_pattern, treeStr, flags=re.UNICODE)
                # "Move up" NER tags when possible
                treeStr = re.sub('\(\s*NP\s+(?P<subtree>\(' + tagname + '(.*)\s*\)\s*\))\s*\)',
                                 '\g<1>', treeStr, flags=re.UNICODE)
        # Add punctuation tags if use_punc_tag is set
        treeStr = self.__parse_punc_tags__(treeStr)
        try:
            parse = Tree.fromstring(treeStr, remove_empty_top_bracketing=True)
            parse = self.__get_nltk_parse_tree__(parse)
        except Exception:
            parse = None
        finally:
            return parse
