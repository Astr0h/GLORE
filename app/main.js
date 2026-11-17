// GLORE desktop shell: opens the window and runs the Python engine beside it.
const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const { spawn, execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const net = require('net');

const REPO_ROOT = path.resolve(__dirname, '..');
const ENGINE_DIR = path.join(REPO_ROOT, 'engine');
let engineProc = null;
let enginePort = 8765;
let logPath = null;

function pythonPath() {
  const win = process.platform === 'win32';
  const venvPy = path.join(ENGINE_DIR, '.venv', win ? 'Scripts' : 'bin', win ? 'python.exe' : 'python');
  if (fs.existsSync(venvPy)) return venvPy;
  return win ? 'python' : 'python3';
}

function freePort(preferred) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once('error', () => {
      const s2 = net.createServer();
      s2.listen(0, '127.0.0.1', () => { const p = s2.address().port; s2.close(() => resolve(p)); });
    });
    srv.listen(preferred, '127.0.0.1', () => srv.close(() => resolve(preferred)));
  });
}

async function startEngine() {
  enginePort = await freePort(8765);
  logPath = path.join(app.getPath('userData'), 'engine.log');
  const log = fs.createWriteStream(logPath, { flags: 'w' });
  const py = pythonPath();
  log.write(`[glore] starting engine: ${py} -m glore_engine --port ${enginePort}\n`);
  engineProc = spawn(py, ['-m', 'glore_engine', '--port', String(enginePort)], {
    cwd: ENGINE_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
    windowsHide: true,
  });
  engineProc.stdout.pipe(log);
  engineProc.stderr.pipe(log);
  engineProc.on('exit', (code) => {
    log.write(`\n[glore] engine exited with code ${code}\n`);
    engineProc = null;
    BrowserWindow.getAllWindows().forEach((w) => w.webContents.send('engine-exit', code));
  });
  engineProc.on('error', (err) => log.write(`\n[glore] failed to start engine: ${err.message}\n`));
}

function stopEngine() {
  if (!engineProc) return;
  if (process.platform === 'win32') {
    execFile('taskkill', ['/pid', String(engineProc.pid), '/T', '/F'], () => {});
  } else {
    engineProc.kill();
  }
  engineProc = null;
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: '#0b0b0d',
    title: 'GLORE',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      additionalArguments: [`--glore-port=${enginePort}`],
    },
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

ipcMain.handle('pick-files', async () => {
  const res = await dialog.showOpenDialog({
    properties: ['openFile', 'multiSelections'],
    filters: [{ name: 'Audio', extensions: ['wav', 'mp3', 'flac', 'aif', 'aiff', 'ogg'] }],
  });
  return res.canceled ? [] : res.filePaths;
});
ipcMain.handle('open-log', () => logPath && shell.openPath(logPath));
ipcMain.handle('restart-engine', async () => { stopEngine(); await startEngine(); return enginePort; });

app.whenReady().then(async () => {
  await startEngine();
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});
app.on('before-quit', stopEngine);
app.on('window-all-closed', () => { stopEngine(); app.quit(); });
