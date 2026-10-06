import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import { v4 as uuid } from 'uuid';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createClient } from '@supabase/supabase-js';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(__dirname, '../data');
const UPLOAD_DIR = path.join(__dirname, '../uploads');
const DB_FILE = path.join(DATA_DIR, 'db.json');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_ANON_KEY || '';
const useSupabase = !!(SUPABASE_URL && SUPABASE_KEY);

let supabase = null;
if (useSupabase) {
  supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
  console.log('Storage: Supabase');
} else {
  console.log('Storage: JSON locale');
}

function loadDb() {
  if (!fs.existsSync(DB_FILE)) {
    const init = { folders: [], notes: [], conversations: [], messages: [], flashcards: [], quizzes: [] };
    fs.writeFileSync(DB_FILE, JSON.stringify(init, null, 2));
    return init;
  }
  return JSON.parse(fs.readFileSync(DB_FILE, 'utf-8'));
}
function saveDb(db) {
  fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2));
}

async function getAll(table) {
  if (useSupabase) {
    const { data, error } = await supabase.from(table).select('*').order('created_at', { ascending: false });
    if (error) throw error;
    return data || [];
  }
  return loadDb()[table] || [];
}

async function getById(table, id) {
  if (useSupabase) {
    const { data, error } = await supabase.from(table).select('*').eq('id', id).single();
    if (error) return null;
    return data;
  }
  return (loadDb()[table] || []).find(x => x.id === id) || null;
}

async function getWhere(table, column, value) {
  if (useSupabase) {
    const { data, error } = await supabase.from(table).select('*').eq(column, value).order('created_at', { ascending: false });
    if (error) throw error;
    return data || [];
  }
  return (loadDb()[table] || []).filter(x => x[column] === value);
}

async function insert(table, row) {
  if (useSupabase) {
    const { data, error } = await supabase.from(table).insert(row).select().single();
    if (error) throw error;
    return data;
  }
  const db = loadDb();
  db[table] = db[table] || [];
  db[table].unshift(row);
  saveDb(db);
  return row;
}

async function update(table, id, updates) {
  if (useSupabase) {
    const { data, error } = await supabase.from(table).update(updates).eq('id', id).select().single();
    if (error) throw error;
    return data;
  }
  const db = loadDb();
  const item = (db[table] || []).find(x => x.id === id);
  if (!item) return null;
  Object.assign(item, updates);
  saveDb(db);
  return item;
}

async function remove(table, id) {
  if (useSupabase) {
    const { error } = await supabase.from(table).delete().eq('id', id);
    if (error) throw error;
    return;
  }
  const db = loadDb();
  db[table] = (db[table] || []).filter(x => x.id !== id);
  saveDb(db);
}

async function removeWhere(table, column, value) {
  if (useSupabase) {
    const { error } = await supabase.from(table).delete().eq(column, value);
    if (error) throw error;
    return;
  }
  const db = loadDb();
  db[table] = (db[table] || []).filter(x => x[column] !== value);
  saveDb(db);
}

const PROVIDER = (process.env.AI_PROVIDER || 'openrouter').toLowerCase();
const OLLAMA_BASE = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'meta-llama/llama-3.2-3b-instruct:free';

async function listModels() {
  try {
    if (PROVIDER === 'ollama') {
      const res = await fetch(`${OLLAMA_BASE}/api/tags`);
      if (!res.ok) throw new Error('Ollama offline');
      const data = await res.json();
      return (data.models || []).map(m => ({ id: m.name, name: m.name }));
    }
    if (PROVIDER === 'openrouter') {
      return [
        { id: 'meta-llama/llama-3.2-3b-instruct:free', name: 'Llama 3.2 3B (free)' },
        { id: 'google/gemma-2-9b-it:free', name: 'Gemma 2 9B (free)' },
        { id: 'mistralai/mistral-7b-instruct:free', name: 'Mistral 7B (free)' },
        { id: 'qwen/qwen-2.5-7b-instruct:free', name: 'Qwen 2.5 7B (free)' },
      ];
    }
    return [{ id: DEFAULT_MODEL, name: DEFAULT_MODEL }];
  } catch {
    return [{ id: DEFAULT_MODEL, name: DEFAULT_MODEL }];
  }
}

async function* streamChat(messages, model = DEFAULT_MODEL) {
  try {
    let url, headers, body;
    if (PROVIDER === 'ollama') {
      url = `${OLLAMA_BASE}/api/chat`;
      headers = { 'Content-Type': 'application/json' };
      body = JSON.stringify({ model, messages, stream: true });
    } else {
      const base = PROVIDER === 'openrouter' ? 'https://openrouter.ai/api/v1' :
                   PROVIDER === 'groq' ? 'https://api.groq.com/openai/v1' :
                   process.env.CUSTOM_BASE_URL || 'https://api.openai.com/v1';
      const key = process.env.OPENROUTER_API_KEY || process.env.GROQ_API_KEY || process.env.OPENAI_API_KEY || '';
      url = `${base}/chat/completions`;
      headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${key}`,
        'HTTP-Referer': process.env.SITE_URL || 'https://kiwi-ai-clone.vercel.app',
        'X-Title': 'Kiwi AI Clone',
      };
      body = JSON.stringify({ model, messages, stream: true });
    }

    const res = await fetch(url, { method: 'POST', headers, body });
    if (!res.ok) {
      const errText = await res.text();
      yield { content: `\n\nErrore AI (${PROVIDER}): ${res.status} – ${errText.slice(0, 300)}`, done: true };
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      if (PROVIDER === 'ollama') {
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const json = JSON.parse(line);
            if (json.message?.content) yield { content: json.message.content, done: false };
            if (json.done) { yield { content: '', done: true }; return; }
          } catch {}
        }
      } else {
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') { yield { content: '', done: true }; return; }
            try {
              const json = JSON.parse(data);
              const content = json.choices?.[0]?.delta?.content || '';
              if (content) yield { content, done: false };
            } catch {}
          }
        }
      }
    }
    yield { content: '', done: true };
  } catch (err) {
    yield { content: `\n\nErrore: ${err.message}`, done: true };
  }
}

async function completeChat(messages, model = DEFAULT_MODEL) {
  let full = '';
  for await (const chunk of streamChat(messages, model)) {
    if (chunk.content) full += chunk.content;
    if (chunk.done) break;
  }
  return full;
}

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors({ origin: true }));
app.use(express.json({ limit: '20mb' }));
app.use('/uploads', express.static(UPLOAD_DIR));

const upload = multer({ dest: UPLOAD_DIR, limits: { fileSize: 20 * 1024 * 1024 } });

app.get('/api/health', (_, res) => res.json({ status: 'ok', storage: useSupabase ? 'supabase' : 'json' }));

app.get('/api/ai/health', async (_, res) => {
  res.json({ ok: true, provider: PROVIDER, message: `${PROVIDER} configurato`, storage: useSupabase ? 'supabase' : 'json' });
});

app.get('/api/ai/models', async (_, res) => {
  const models = await listModels();
  res.json({ provider: PROVIDER, models });
});

app.get('/api/folders', async (_, res) => {
  try { res.json(await getAll('folders')); } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/folders', async (req, res) => {
  try {
    const folder = await insert('folders', { id: uuid(), name: req.body.name || 'Nuova cartella', created_at: new Date().toISOString() });
    res.status(201).json(folder);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.delete('/api/folders/:id', async (req, res) => {
  try { await remove('folders', req.params.id); res.json({ ok: true }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/notes', async (req, res) => {
  try {
    let notes = await getAll('notes');
    if (req.query.folder_id) notes = notes.filter(n => n.folder_id === req.query.folder_id);
    res.json(notes);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/notes/:id', async (req, res) => {
  try {
    const note = await getById('notes', req.params.id);
    if (!note) return res.status(404).json({ error: 'Nota non trovata' });
    const flashcards = await getWhere('flashcards', 'note_id', note.id);
    const quizzes = await getWhere('quizzes', 'note_id', note.id);
    res.json({ ...note, flashcards, quizzes });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/notes', async (req, res) => {
  try {
    const { title, content, source_type, source_name, source_text, folder_id, generate } = req.body;
    let finalContent = content || '';
    let finalTitle = title || source_name || 'Nuova nota';

    if (generate && source_text) {
      try {
        finalContent = await completeChat([
          { role: 'system', content: 'Sei un tutor AI per studenti. Crea note di studio chiare in Markdown. Solo le note, niente introduzioni.' },
          { role: 'user', content: `Materiale:\n\n${String(source_text).slice(0, 10000)}\n\nGenera note strutturate.` }
        ]);
        if (!title) {
          const t = await completeChat([
            { role: 'system', content: 'Rispondi solo con un titolo breve (max 8 parole).' },
            { role: 'user', content: finalContent.slice(0, 400) }
          ]);
          finalTitle = t.replace(/["\n]/g, '').trim().slice(0, 80) || finalTitle;
        }
      } catch (e) {
        finalContent = `# Note\n\nErrore generazione: ${e.message}\n\n${String(source_text).slice(0, 2000)}`;
      }
    }

    const note = await insert('notes', {
      id: uuid(),