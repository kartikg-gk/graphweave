import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import ts from 'typescript';
const source = fs.readFileSync(new URL('../src/lib/api.ts', import.meta.url), 'utf8')
  .replace('import { getAuthToken } from "@/lib/authToken";', 'const getAuthToken = async () => null;')
  .replace('import { layoutGraph } from "@/lib/layout";', 'const layoutGraph = (nodes, edges) => ({nodes, edges});')
  .replaceAll('import.meta.env', '({})');
const js = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText;
const api = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'));
const response = (events) => new Response(events.map(e => JSON.stringify(e) + '\n').join(''));
test('complete streams cache by actual context; partial streams are never cached', async () => {
  let calls = 0;
  globalThis.fetch = async () => { calls++; return response([{type:'token',text:'saved'}, {type:'done'}]); };
  assert.equal(await api.streamAnswer('q', 'abc', () => {}), 'saved');
  assert.equal(await api.streamAnswer('q', 'abc', () => {}), 'saved');
  assert.equal(calls, 1);
  await api.streamAnswer('q', 'xyz', () => {});
  assert.equal(calls, 2);
  globalThis.fetch = async () => { calls++; return response([{type:'token',text:'partial'}]); };
  await assert.rejects(api.streamAnswer('broken', 'ctx', () => {}), /before completion/);
  await assert.rejects(api.streamAnswer('broken', 'ctx', () => {}), /before completion/);
  assert.equal(calls, 4);
  globalThis.fetch = async () => response([{type:'token',text:'partial'}, {type:'error',text:'failed'}]);
  await assert.rejects(api.streamAnswer('error', 'ctx', () => {}), /failed/);
});
test('answer save targets the existing trace and surfaces storage failure', async () => {
  globalThis.fetch = async (url, init) => {
    assert.ok(url.endsWith('/api/traces/id%2Fone/answer'));
    assert.equal(init.method, 'PUT');
    assert.deepEqual(JSON.parse(init.body), {answer:'final'});
    return new Response('{}', {status:500});
  };
  await assert.rejects(api.saveTraceAnswer('id/one','final'));
});
