"""GLORE engine - local HTTP API the desktop app talks to.
Binds to 127.0.0.1 only; nothing leaves your machine."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import __version__
from .analysis import analyse, parse_key, key_name, camelot
from .audio_io import SUPPORTED_EXTENSIONS, load_stereo, save_wav, duration_seconds
from .project import ProjectStore, Source, Layer, JobRunner, KINDS, PACKS, new_id
from . import separation, cleanup, linkimport, datasets, tracklist, cloud, training, generate as gen
from .cloud import CloudError, CloudManager
from .vocal_lock import lock_vocal


# ---------------------------------------------------------------- request bodies

class ProjectPatch(BaseModel):
    name: str | None = None
    bpm: float | None = None
    key: str | None = None
    genre_pack: str | None = None
    monitor_mode: str | None = None
    step: str | None = None
    follow_vocal: bool | None = None  # True = project BPM goes back to following the vocal


class ImportBody(BaseModel):
    path: str
    kind: str | None = None  # if omitted, GLORE guesses


class KindBody(BaseModel):
    kind: str


class OverrideBody(BaseModel):
    bpm: float | None = None
    key: str | None = None
    clear: bool = False


class LinkBody(BaseModel):
    url: str
    kind: str | None = None


class RawBody(BaseModel):
    stages: list[str]


class TagBody(BaseModel):
    pack: str | None = None  # None = remove from training sets
    moods: list[str] = []


class TrackListBody(BaseModel):
    csv_path: str | None = None   # CSV exported from the track-list sheet
    csv_text: str | None = None


class CloudSettingsBody(BaseModel):
    api_key: str | None = None
    s3_access_key: str | None = None
    s3_secret: str | None = None
    datacenter: str | None = None
    volume_id: str | None = None
    volume_gb: int | None = None
    idle_minutes: float | None = None
    hard_cap_hours: float | None = None
    monthly_limit: float | None = None
    gpu: str | None = None
    variant: str | None = None
    image: str | None = None


class GenerateBody(BaseModel):
    pack: str = "house"
    lora_run: str | None = None   # default: the pack's newest LoRA
    use_lora: bool = True
    caption: str | None = None    # default: pack name (+ moods)
    bpm: float | None = None      # default: project BPM
    key: str | None = None        # default: project key
    duration: float = 60.0
    seed: int = -1
    count: int = 1
    scale: float = 1.0


class AroundVocalBody(BaseModel):
    source_id: str | None = None   # default: the locked vocal, else the newest vocal
    caption: str = "soft house"
    tracks: list[str] | None = None
    seed: int = -1
    steps: int = 50
    guidance: float = 7.0
    beat_gain_db: float = -3.0
    export_dir: str | None = None  # also write an MP3 of the mix here


class TrainBody(BaseModel):
    epochs: int | None = None   # default: picked from the track count
    rank: int | None = None     # default 64


class LockBody(BaseModel):
    source_id: str
    target_bpm: float | None = None  # defaults to project BPM
    pitch_semitones: float = 0.0



def create_app(store: ProjectStore | None = None, cloud_manager: CloudManager | None = None,
               resume_cloud: bool = True) -> FastAPI:
    app = FastAPI(title="GLORE engine", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.store = store or ProjectStore()
    app.state.jobs = JobRunner()
    app.state.cloud = cloud_manager or CloudManager()
    app.state.cloud_jobs = {}  # "train:<pack>" / "fit:<pack>" / "test" / "volume" -> engine job id

    def S() -> ProjectStore:
        return app.state.store

    def get_source(source_id: str) -> Source:
        try:
            return S().source(source_id)
        except KeyError:
            raise HTTPException(404, "Sample not found")

    def project_payload():
        p = S().project.to_dict()
        p["root"] = str(S().root)
        return p

    # ------------------------------------------------------------------ meta
    @app.get("/health")
    def health():
        return {"ok": True, "version": __version__, "gpu": separation.device_info()}

    @app.get("/project")
    def get_project():
        return project_payload()

    @app.post("/project/new")
    def new_project():
        S().reset()
        return project_payload()


    @app.patch("/project")
    def patch_project(body: ProjectPatch):
        p = S().project
        data = body.model_dump(exclude_none=True)
        if data.pop("follow_vocal", None):
            p.bpm_locked_by_user = False
            _maybe_follow_vocal_tempo()
        if "bpm" in data:
            if not 40 <= data["bpm"] <= 220:
                raise HTTPException(400, "BPM must be between 40 and 220")
            p.bpm_locked_by_user = True
        if "genre_pack" in data and data["genre_pack"] not in PACKS:
            raise HTTPException(400, f"genre_pack must be one of {list(PACKS)}")
        if "monitor_mode" in data and data["monitor_mode"] not in ("headphones", "sound_system"):
            raise HTTPException(400, "monitor_mode must be headphones or sound_system")
        for k, v in data.items():
            setattr(p, k, v)
        S().save()
        return project_payload()

    # --------------------------------------------------------------- import

    @app.post("/sources/import")
    def import_source(body: ImportBody):
        src_path = Path(body.path)
        if not src_path.exists():
            raise HTTPException(400, f"File not found: {src_path}")
        if src_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type {src_path.suffix}. Use WAV, MP3, FLAC, AIFF, or OGG.")
        if body.kind and body.kind not in KINDS:
            raise HTTPException(400, f"kind must be one of {KINDS}")
        try:
            src = add_audio_source(src_path, src_path.stem, body.kind, str(src_path), "file")
        except Exception as e:
            raise HTTPException(400, f"Couldn't read that file: {e}")
        # Analyse straight away so the tag guess + BPM/key arrive automatically
        job = _start_analysis(src, auto_kind=body.kind is None)
        return {"source": asdict(src), "job_id": job.id}

    def add_audio_source(path: Path, name: str, kind: str | None, original: str, origin: str,
                         derived_from: str | None = None) -> Source:
        """Copy any audio into the project as 44.1k stereo WAV and register it."""
        sid = new_id()
        audio = load_stereo(path)
        fname = f"{sid}.wav"
        save_wav(S().sources_dir / fname, audio)
        src = Source(id=sid, name=name, kind=kind or "full_song", file=fname, original_path=original,
                     duration=round(duration_seconds(audio), 3), origin=origin, derived_from=derived_from)
        S().add_source(src)
        return src

    # ------------------------------------------------------------ link import
    @app.post("/sources/import-link")
    def import_link(body: LinkBody):
        try:
            service = linkimport.detect_service(body.url)
        except linkimport.LinkError as e:
            raise HTTPException(400, str(e))
        if body.kind and body.kind not in KINDS:
            raise HTTPException(400, f"kind must be one of {KINDS}")

        def work(report):
            wav, info, tmp = linkimport.fetch_link(body.url, progress=report)
            try:
                name = linkimport.safe_name(f"{info.artist} - {info.title}" if info.artist else info.title)
                report(0.9, "Adding to project")
                src = add_audio_source(wav, name, body.kind, body.url, "link")
            finally:
                tmp.cleanup()
            _start_analysis(src, auto_kind=body.kind is None)
            return {"source_id": src.id, "service": service, "title": info.title,
                    "artist": info.artist, "matched": info.matched}

        return {"job_id": app.state.jobs.start("import_link", work).id, "service": service}

    # -------------------------------------------------------------- raw vocal
    @app.get("/raw-vocal/stages")
    def raw_stages():
        return [{"id": k, "label": cleanup.STAGE_LABELS[k]} for k in cleanup.STAGE_ORDER]

    @app.post("/sources/{source_id}/raw-vocal")
    def raw_vocal(source_id: str, body: RawBody):
        src = get_source(source_id)
        if src.kind != "vocal":
            raise HTTPException(400, "Raw vocal works on samples tagged Vocal")
        bad = [s for s in body.stages if s not in cleanup.STAGES]
        if bad or not body.stages:
            raise HTTPException(400, f"Choose from {cleanup.STAGE_ORDER}")
        in_path = S().sources_dir / src.file

        def work(report):
            out = S().renders_dir / f"raw_{src.id}.wav"
            cleanup.raw_vocal(in_path, out, body.stages, progress=lambda p, m: report(p * 0.92, m))
            raw = add_audio_source(out, f"{src.name} (raw)", "vocal", src.original_path, "raw",
                                   derived_from=src.derived_from or src.id)
            raw.overrides = dict(src.overrides)  # keep any tempo/key the user corrected
            S().save()
            _start_analysis(raw, auto_kind=False)
            return {"source_id": raw.id, "stages": body.stages}

        return {"job_id": app.state.jobs.start("raw_vocal", work, gpu=True).id}

    # ------------------------------------------------- genre tags / datasets
    @app.get("/packs")
    def packs():
        out = datasets.summary()
        for p in out:
            p["loras"] = [{k: l.get(k) for k in ("run_id", "trained_at", "tracks", "epochs", "loss", "variant", "path")}
                          for l in training.list_loras(p["id"])]
        return out

    @app.post("/sources/{source_id}/tag")
    def tag_source(source_id: str, body: TagBody):
        src = get_source(source_id)
        moods = sorted({m.strip() for m in body.moods if m.strip()}, key=str.lower)[:12]
        datasets.remove_by_source(f"{S().root.name}:{src.id}")
        if body.pack:
            if body.pack not in PACKS:
                raise HTTPException(400, f"pack must be one of {list(PACKS)}")
            eff = src.effective()
            datasets.add_or_update(body.pack, S().sources_dir / src.file, src.name, moods,
                                   bpm=eff.get("bpm"), key=eff.get("key"),
                                   source_ref=f"{S().root.name}:{src.id}")
        src.pack, src.moods = body.pack, moods
        S().save()
        return {"source": asdict(src), "packs": datasets.summary()}

    @app.get("/datasets/{pack}")
    def dataset_items(pack: str):
        if pack not in PACKS:
            raise HTTPException(404, "Unknown pack")
        return datasets.list_items(pack)

    @app.delete("/datasets/{pack}/{item_id}")
    def dataset_remove(pack: str, item_id: str):
        if pack not in PACKS:
            raise HTTPException(404, "Unknown pack")
        datasets.remove(pack, item_id)
        for s in S().project.sources:
            if s.pack == pack and not datasets.find_by_source(f"{S().root.name}:{s.id}"):
                s.pack, s.moods = None, []
        S().save()
        return datasets.summary()

    @app.post("/datasets/{pack}/import-list")
    def dataset_import_list(pack: str, body: TrackListBody):
        if pack not in PACKS:
            raise HTTPException(404, "Unknown pack")
        try:
            text = body.csv_text if body.csv_text is not None else Path(body.csv_path or "").read_text(encoding="utf-8-sig")
            rows = tracklist.parse_csv(text)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            raise HTTPException(400, f"Couldn't read that track list: {e}")
        if not rows:
            raise HTTPException(400, "No ticked tracks in that list")
        job = app.state.jobs.start("import_list", lambda r: tracklist.import_rows(pack, rows, r).to_dict())
        return {"job_id": job.id, "tracks": len(rows)}

    @app.post("/datasets/{pack}/prepare")
    def dataset_prepare(pack: str):
        if pack not in PACKS:
            raise HTTPException(404, "Unknown pack")
        return {"job_id": app.state.jobs.start("prepare", lambda r: datasets.prepare(pack, r), gpu=True).id}


    @app.post("/sources/{source_id}/kind")
    def set_kind(source_id: str, body: KindBody):
        if body.kind not in KINDS:
            raise HTTPException(400, f"kind must be one of {KINDS}")
        src = get_source(source_id)
        src.kind = body.kind
        S().save()
        job = _start_analysis(src, auto_kind=False)  # phrases only exist for vocals - re-run
        return {"source": asdict(src), "job_id": job.id}

    @app.delete("/sources/{source_id}")
    def delete_source(source_id: str):
        get_source(source_id)
        S().remove_source(source_id)
        return project_payload()

    # -------------------------------------------------------------- analyse
    def _start_analysis(src: Source, auto_kind: bool):
        path = str(S().sources_dir / src.file)

        def work(report):
            report(0.1, "Analysing tempo, key and phrases")
            result = analyse(path, kind=None if auto_kind else src.kind)
            if auto_kind:
                src.kind = result.kind_guess
                if result.kind_guess == "vocal":
                    result = analyse(path, kind="vocal")  # get phrases
            # A vocal extracted from a full song inherits that song's tempo/key:
            # the full mix (with drums) gives a far more reliable reading.
            if src.derived_from:
                try:
                    parent = S().source(src.derived_from).effective()
                    for k in ("bpm", "key", "key_tonic", "key_mode", "camelot", "beats", "downbeats"):
                        if k in parent:
                            setattr(result, k, parent[k])
                except KeyError:
                    pass
            src.analysis = result.to_dict()
            _maybe_follow_vocal_tempo()
            S().save()
            return {"source_id": src.id, "analysis": src.analysis, "kind": src.kind}

        return app.state.jobs.start("analyse", work)

    @app.post("/sources/{source_id}/analyse")
    def reanalyse(source_id: str):
        src = get_source(source_id)
        return {"job_id": _start_analysis(src, auto_kind=False).id}


    @app.patch("/sources/{source_id}/analysis")
    def override_analysis(source_id: str, body: OverrideBody):
        src = get_source(source_id)
        if body.clear:
            src.overrides = {}
        if body.bpm is not None:
            if not 40 <= body.bpm <= 220:
                raise HTTPException(400, "BPM must be between 40 and 220")
            src.overrides["bpm"] = round(body.bpm, 3)
        if body.key:
            try:
                tonic, mode = parse_key(body.key)
            except ValueError as e:
                raise HTTPException(400, str(e))
            src.overrides.update(key=key_name(tonic, mode), key_tonic=tonic, key_mode=mode,
                                 camelot=camelot(tonic, mode))
        _maybe_follow_vocal_tempo()
        S().save()
        return {"source": asdict(src), "effective": src.effective()}

    def _maybe_follow_vocal_tempo():
        """Project tempo follows the vocal's natural tempo until the user sets it."""
        p = S().project
        if p.bpm_locked_by_user:
            return
        vocals = [s for s in p.sources if s.kind == "vocal" and s.analysis]
        if vocals:
            eff = vocals[0].effective()
            p.bpm = eff.get("bpm")
            p.key = eff.get("key")

    # -------------------------------------------------------- extract vocal
    @app.post("/sources/{source_id}/extract-vocal")
    def extract_vocal(source_id: str):
        src = get_source(source_id)
        in_path = S().sources_dir / src.file
        out_dir = S().stems_dir / src.id

        def work(report):
            stems = separation.separate(in_path, out_dir, progress=lambda p, m: report(p * 0.9, m))
            for name, p in stems.items():
                S().upsert_layer(Layer(id=f"stem-{src.id}-{name}", name=f"{src.name} · {name}",
                                       file=str(p.relative_to(S().root)).replace("\\", "/"),
                                       role="stem", source_id=src.id))
            # The vocal stem becomes a new sample in its own right
            vid = new_id()
            vfile = f"{vid}.wav"
            (S().sources_dir / vfile).write_bytes(stems["vocals"].read_bytes())
            vocal = Source(id=vid, name=f"{src.name} (vocal)", kind="vocal", file=vfile,
                           original_path=src.original_path, duration=src.duration, derived_from=src.id)
            S().add_source(vocal)
            report(0.92, "Analysing extracted vocal")
            _start_analysis(vocal, auto_kind=False)
            S().save()
            return {"vocal_source_id": vid, "stems": list(stems)}

        return {"job_id": app.state.jobs.start("extract_vocal", work, gpu=True).id}

    # ----------------------------------------------------------- vocal lock

    @app.post("/vocal-lock")
    def vocal_lock(body: LockBody):
        src = get_source(body.source_id)
        if src.kind != "vocal":
            raise HTTPException(400, "Vocal lock needs a sample tagged as Vocal")
        eff = src.effective()
        if not eff.get("bpm"):
            raise HTTPException(400, "Analyse the vocal first")
        target = body.target_bpm or S().project.bpm or eff["bpm"]
        out_rel = "renders/vocal_locked.wav"
        ratio = target / eff["bpm"]
        if not 0.5 <= ratio <= 2.0:
            raise HTTPException(400, f"{eff['bpm']:g} → {target:g} BPM is too big a jump for a clean "
                                     "stretch. Try half or double the target BPM.")

        def work(report):
            report(0.2, f"Stretching {eff['bpm']:g} → {target:g} BPM")
            res = lock_vocal(S().sources_dir / src.file, S().root / out_rel, source_bpm=eff["bpm"],
                             target_bpm=target, pitch_semitones=body.pitch_semitones,
                             phrases=eff.get("phrases") or [])
            S().upsert_layer(Layer(id="vocal-locked", name="Vocal (locked)", file=out_rel,
                                   role="vocal_locked", source_id=src.id))
            S().project.vocal_lock = {"source_id": src.id, **res.to_dict(), "file": out_rel}
            S().save()
            return S().project.vocal_lock

        return {"job_id": app.state.jobs.start("vocal_lock", work).id}


    # ------------------------------------------------------------ cloud GPU
    def _cloud_job(key: str, kind: str, fn, gpu: bool = False):
        cur = app.state.cloud_jobs.get(key)
        if cur and app.state.jobs.jobs.get(cur) and app.state.jobs.jobs[cur].status == "running":
            raise HTTPException(409, "Already running")

        def work(report):
            try:
                return fn(report)
            except CloudError as e:
                raise RuntimeError(str(e)) from None

        job = app.state.jobs.start(kind, work, gpu=gpu)
        app.state.cloud_jobs[key] = job.id
        return {"job_id": job.id}

    def _cloud_errors(fn):
        try:
            return fn()
        except CloudError as e:
            raise HTTPException(400, str(e))

    @app.get("/cloud/settings")
    def cloud_settings():
        return cloud.public_settings()

    @app.put("/cloud/settings")
    def cloud_settings_put(body: CloudSettingsBody):
        return _cloud_errors(lambda: cloud.public_settings(cloud.save_settings(body.model_dump(exclude_none=True))))

    @app.get("/cloud/status")
    def cloud_status():
        st = _cloud_errors(app.state.cloud.status)
        st["jobs"] = {k: v for k, v in app.state.cloud_jobs.items()
                      if app.state.jobs.jobs.get(v) and app.state.jobs.jobs[v].status == "running"}
        return st

    @app.get("/cloud/quote")
    def cloud_quote():
        """What a training run would use and could cost at most, before you press go."""
        def q():
            s = cloud.load_settings()
            if cloud.missing(s):
                raise CloudError("Finish the cloud settings first")
            gpu, price = cloud.pick_gpu(app.state.cloud.rp().gpus(), s["datacenter"], s["gpu"])
            return {"gpu": gpu, "price_per_hr": price, "hard_cap_hours": s["hard_cap_hours"],
                    "max_cost": round(price * float(s["hard_cap_hours"]), 2),
                    "month_spend": app.state.cloud.month_spend(), "monthly_limit": s["monthly_limit"]}
        return _cloud_errors(q)

    @app.post("/cloud/volume")
    def cloud_volume():
        return _cloud_job("volume", "cloud_volume", lambda r: training.setup_volume(app.state.cloud, r))

    @app.post("/cloud/test")
    def cloud_test():
        return _cloud_job("test", "cloud_test", lambda r: training.connection_test(app.state.cloud, r))

    @app.post("/cloud/stop")
    def cloud_stop():
        return {"stopped": _cloud_errors(app.state.cloud.stop_all)}

    @app.post("/generate")
    def generate_ep(body: GenerateBody):
        if body.pack not in PACKS:
            raise HTTPException(404, "Unknown pack")
        if not 10 <= body.duration <= 480:
            raise HTTPException(400, "Length must be 10-480 seconds")
        if not 1 <= body.count <= 4:
            raise HTTPException(400, "1-4 takes")
        if not 0 <= body.scale <= 1.5:
            raise HTTPException(400, "LoRA strength must be 0-1.5")
        if body.bpm is not None and not 30 <= body.bpm <= 300:
            raise HTTPException(400, "BPM must be 30-300")

        def work(report):
            try:
                return gen.generate(S(), report, **body.model_dump())
            except gen.GenerateError as e:
                raise RuntimeError(str(e)) from None

        return _cloud_job("generate", "generate", work, gpu=True)

    @app.post("/generate/around-vocal")
    def generate_around(body: AroundVocalBody):
        if not 8 <= body.steps <= 100:
            raise HTTPException(400, "Steps must be 8-100")

        def work(report):
            try:
                return gen.generate_around_vocal(S(), report, **body.model_dump())
            except gen.GenerateError as e:
                raise RuntimeError(str(e)) from None

        return _cloud_job("generate", "generate_around_vocal", work, gpu=True)

    @app.get("/logs/{name}")
    def log_tail(name: str, lines: int = 80):
        """End of a GLORE log, for the app's log viewer and for remote help."""
        home = training.glore_home()
        files = {"fit-test": home / "fit-test" / "log.txt", "ace-install": home / "ace-install.log"}
        if name == "generate":
            logs = sorted((S().root / "generated").glob("*.log"), key=lambda f: f.stat().st_mtime)
            path = logs[-1] if logs else None
        elif name.startswith("cloud-"):
            path = cloud.home() / f"log-{name[6:]}.txt"
        else:
            path = files.get(name)
        if not path or not path.exists():
            raise HTTPException(404, "No such log yet")
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"path": str(path), "text": "\n".join(text[-max(1, min(lines, 2000)):])}

    @app.get("/loras")
    def loras():
        return training.list_loras()

    @app.post("/datasets/{pack}/train")
    def dataset_train(pack: str, body: TrainBody | None = None):
        if pack not in PACKS:
            raise HTTPException(404, "Unknown pack")
        b = body or TrainBody()
        if b.epochs is not None and not 1 <= b.epochs <= 2000:
            raise HTTPException(400, "Epochs must be between 1 and 2000")
        if b.rank is not None and b.rank not in (8, 16, 32, 64, 128):
            raise HTTPException(400, "Rank must be 8, 16, 32, 64 or 128")
        if cloud.missing(cloud.load_settings()):
            raise HTTPException(400, "Finish the cloud settings first")
        return _cloud_job(f"train:{pack}", "train_lora",
                          lambda r: training.train_cloud(pack, app.state.cloud, r, b.epochs, b.rank))

    @app.post("/datasets/{pack}/fit-test")
    def dataset_fit_test(pack: str):
        if pack not in PACKS:
            raise HTTPException(404, "Unknown pack")
        return _cloud_job(f"fit:{pack}", "fit_test", lambda r: training.fit_test(pack, r), gpu=True)

    def _resume_cloud():
        """After a restart: keep the safety net on any pods still up, and pick up runs in flight."""
        if cloud.missing(cloud.load_settings()):
            return
        try:
            app.state.cloud.resume_watching()
        except Exception:
            pass
        for job_id, run in training.pending_runs().items():
            _cloud_job(f"train:{run['pack']}", "train_lora",
                       lambda r, j=job_id: training.follow(j, app.state.cloud, r))

    if resume_cloud:
        _resume_cloud()

    # ----------------------------------------------------------------- jobs
    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        job = app.state.jobs.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return asdict(job)

    # ---------------------------------------------------------------- audio
    @app.get("/audio/{rel_path:path}")
    def audio(rel_path: str):
        try:
            p = S().path_of(rel_path)
        except ValueError:
            raise HTTPException(403, "Not allowed")
        if not p.exists():
            raise HTTPException(404, "Audio not found")
        return FileResponse(p, media_type="audio/wav")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--project-root", type=str, default=None)
    args = ap.parse_args()
    store = ProjectStore(Path(args.project_root) if args.project_root else None)
    uvicorn.run(create_app(store), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
