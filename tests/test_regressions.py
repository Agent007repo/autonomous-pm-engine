import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
ROOT = Path(__file__).resolve().parents[1]

def definitions(filename, names, namespace):
    path = ROOT / filename
    if path.suffix == '.ipynb':
        nb = json.loads(path.read_text())
        text = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        text = '\n'.join(line if not line.startswith(('!', '%', 'pip install')) else '# '+line for line in text.splitlines())
    else:
        text = path.read_text()
    tree = ast.parse(text)
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    body += [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace
import asyncio, tempfile, hashlib, re, math
from collections import Counter
from typing import Any
import numpy as np
from upload_safety import safe_upload_name, bounded_upload

class Document:
    def __init__(self,page_content,metadata=None):
        self.page_content=page_content;self.metadata=metadata or {}

class Logger:
    def __getattr__(self,name): return lambda *args,**kw:None

VS=definitions('vector_store.py',{'VectorStore'},
    dict(hashlib=hashlib,json=json,math=math,re=re,Counter=Counter,Document=Document,logger=Logger()))['VectorStore']
GS=definitions('graph_store.py',{'GraphStore'},dict(logger=Logger(),json=json,re=re))['GraphStore']
SC=definitions('semantic_chunker.py',{'SemanticChunker'},dict(np=np,re=re,Document=Document,logger=Logger()))['SemanticChunker']

class PMRegressionTests(unittest.TestCase):
    def test_prompt_literal_json_does_not_break_format(self):
        self.assertIn('CHUNK',GS._EXTRACTION_PROMPT.format(doc_type='survey',source='x',text='feedback'))
    def test_document_identity_preserves_distinct_sources(self):
        a=Document('same',{'source':'a'});b=Document('same',{'source':'b'})
        self.assertNotEqual(VS._doc_id(a),VS._doc_id(b))
        self.assertEqual(VS._doc_id(a),VS._doc_id(Document('same',{'source':'a','_distance':.2})))
    def test_lexical_search_never_invokes_default_embedder(self):
        store=object.__new__(VS);store._cfg=SimpleNamespace(top_k_retrieval=10)
        store._collection=MagicMock();store._collection.get.return_value={
            'documents':['search latency slow','onboarding tutorials','search search'],
            'metadatas':[{}, {}, {}]}
        result=store.keyword_search('search')
        self.assertEqual(len(result),2)
        self.assertFalse(store._collection.query.called)
    def test_pure_sparse_does_not_call_dense_search(self):
        store=object.__new__(VS);store._cfg=SimpleNamespace(top_k_retrieval=10,hybrid_alpha=.7)
        store.similarity_search=MagicMock();store.keyword_search=MagicMock(return_value=[Document('word')])
        self.assertEqual(len(store.hybrid_search('word',alpha=0)),1)
        self.assertFalse(store.similarity_search.called)
    def test_centroid_cosine_is_normalized(self):
        chunker=object.__new__(SC);chunker.window_size=2;chunker.threshold=.8
        embedding=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.],[1.,0.]])
        self.assertEqual(chunker._find_boundaries(embedding),[])
    def test_long_single_sentence_respects_word_budget(self):
        chunker=object.__new__(SC);chunker.max_chunk_tokens=64
        chunks=chunker._bisect_chunk([' '.join(['word']*1000)])
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(len(' '.join(c).split())*1.3<=64 for c in chunks))
    def test_upload_traversal_and_unsupported_files_rejected(self):
        for name in ['../secret.txt','/tmp/secret.txt','..\\secret.txt','payload.exe','x:secret.txt']:
            with self.assertRaises(ValueError): safe_upload_name(name)
        self.assertEqual(safe_upload_name('feedback.txt'),'feedback.txt')
    def test_upload_size_enforced_during_stream(self):
        class Upload:
            def __init__(self): self.data=b'x'*11
            async def read(self,n):
                out,self.data=self.data[:n],self.data[n:];return out
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): asyncio.run(bounded_upload(Upload(),Path(d)/'file.txt',10))
    def test_transient_document_channels_are_declared(self):
        ns=definitions('state.py',{'PipelineState'},{'TypedDict':__import__('typing').TypedDict})
        self.assertTrue({'_raw_documents','_chunks'}<=set(ns['PipelineState'].__annotations__))

class NodeRegressionTests(unittest.TestCase):
    def test_embed_uses_bound_dependencies_and_declared_chunks(self):
        ns=definitions('nodes.py',{'embed_node'},dict(logger=Logger()))
        vs=MagicMock();vs.upsert_documents.return_value=1
        chunker=MagicMock();chunker.chunk_documents.return_value=[Document('chunk')]
        result=ns['embed_node']({'_raw_documents':[Document('raw')]},vector_store=vs,chunker=chunker)
        self.assertEqual(result['chunk_count'],1);self.assertEqual(len(result['_chunks']),1)
        self.assertFalse(ns['embed_node']({'errors':['failed']},vector_store=vs,chunker=chunker))
    def test_fragment_metadata_keeps_original_sentence_index(self):
        chunker=object.__new__(SC);chunker.max_chunk_tokens=64;chunker.window_size=3
        chunker._embed_sentences=lambda sentences:np.ones((len(sentences),2))
        chunker._find_boundaries=lambda x:[]
        chunks=chunker._chunk_single(Document(' '.join(['word']*200)))
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(c.metadata['sentence_start']==0 and c.metadata['sentence_end']==0 for c in chunks))
