import fs from 'node:fs';

const [, , commitMessageFile] = process.argv;
if (!commitMessageFile) process.exit(0);

const text = fs.readFileSync(commitMessageFile, 'utf8');
const match = text.match(/(?:^|\s)release\s*[:=]\s*(v?\d+\.\d+\.\d+|major|minor|patch)/i);

const bumpMode = match ? match[1].toLowerCase() : 'patch';
const pkgPath = 'package.json';
const lockPath = 'package-lock.json';

const semver = /^v?(\d+)\.(\d+)\.(\d+)$/;

function bumpVersion(version, mode) {
  const m = version.match(semver);
  if (!m) throw new Error(`Bad current version: ${version}`);
  let [_, major, minor, patch] = m;
  let majorN = Number.parseInt(major, 10);
  let minorN = Number.parseInt(minor, 10);
  let patchN = Number.parseInt(patch, 10);

  if (mode === 'major') {
    majorN += 1;
    minorN = 0;
    patchN = 0;
  } else if (mode === 'minor') {
    minorN += 1;
    patchN = 0;
  } else if (mode === 'patch') {
    patchN += 1;
  } else if (semver.test(mode)) {
    return mode.replace(/^v/, '');
  } else {
    throw new Error(`Unsupported release specifier: ${mode}`);
  }

  return `${majorN}.${minorN}.${patchN}`;
}

const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
const lock = JSON.parse(fs.readFileSync(lockPath, 'utf8'));

const current = pkg.version;
const next = bumpVersion(current, bumpMode);
if (next === current) process.exit(0);

pkg.version = next;
lock.version = next;
if (lock.packages && lock.packages['']) {
  lock.packages[''].version = next;
}

fs.writeFileSync(pkgPath, `${JSON.stringify(pkg, null, 2)}\n`);
fs.writeFileSync(lockPath, `${JSON.stringify(lock, null, 2)}\n`);

console.log(`[version] ${current} -> ${next} (mode=${bumpMode})`);
