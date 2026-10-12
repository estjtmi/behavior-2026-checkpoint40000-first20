"""One official rollout per instance, preserving all raw evaluation outputs."""
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path('/workspace/group-b-run')
REPO = Path('/workspace/BEHAVIOR-1K')
PYTHON = '/workspace/envs/behavior/bin/python'
TASKS = [(7, 'picking_up_toys'), (11, 'putting_dishes_away_after_cleaning'),
         (14, 'carrying_in_groceries'), (15, 'bringing_in_wood'),
         (16, 'moving_boxes_to_storage')]
PROVENANCE = {
    'checkpoint': 'estjtmi/behavior-2026-pi05-15tasks/checkpoint_40000',
    'model_revision': '479fd5240a5150d69903506d01c84cd77d25bbd2',
    'training_revision': 'be82c0229384aeb4c64ac0eac1742ab71e4a3647',
    'evaluator_release': 'v3.9.3-post1',
    'evaluator_revision': 'bd049de3119acdcdf2334fe9e1ebe060fa20c108',
    'wrapper': 'omnigibson.eval.wrappers.DefaultWrapper',
    'camera_resolution': [224, 224], 'num_envs': 1, 'num_rollouts': 1,
    'mode': 'public_test', 'cuda_device': 0,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(**status):
    path = ROOT / 'progress.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'updated_at': now(), **status}, indent=2) + '\n')
    tmp.replace(path)


def validate(output, task, index):
    scores = list((output / 'json').glob('*.json'))
    videos = list((output / 'videos').glob('*.mp4'))
    assert len(scores) == len(videos) == 1, (scores, videos)
    result = json.loads(scores[0].read_text())
    assert result['task'] == task and result['instance_id'] == 301 + index
    assert result['rollout_id'] == 0 and isinstance(result['success'], bool)
    q = result['q_score']['final']
    assert isinstance(q, (int, float)) and math.isfinite(q) and 0 <= q <= 1
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,nb_frames:format=duration', '-of', 'json', str(videos[0])]))
    assert info['streams'] and float(info['format']['duration']) > 0, info
    return {'task': task, 'public_index': index, 'instance_id': result['instance_id'],
            'q_score': q, 'success': result['success'], 'steps': result['steps'],
            'score_path': str(scores[0]), 'video_path': str(videos[0]),
            'score_sha256': hashlib.sha256(scores[0].read_bytes()).hexdigest(),
            'video_sha256': hashlib.sha256(videos[0].read_bytes()).hexdigest()}


def summarize(completed):
    tasks = []
    for task_id, name in TASKS:
        rows = [row for row in completed if row['task'] == name]
        tasks.append({'task_id': task_id, 'task': name, 'completed': len(rows),
                      'mean_q_score': sum(row['q_score'] for row in rows) / len(rows) if rows else None,
                      'binary_success_rate': sum(row['success'] for row in rows) / len(rows) if rows else None,
                      'complete': len(rows) == 10, 'artifacts': str(ROOT / 'results' / name)})
    summary = {'updated_at': now(), 'provenance': PROVENANCE, 'tasks': tasks,
               'completed_rollouts': len(completed),
               'group_b_mean_q_score': sum(row['q_score'] for row in completed) / 50 if len(completed) == 50 else None,
               'rollouts': completed}
    path = ROOT / 'summary.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2) + '\n')
    tmp.replace(path)
    return tasks


def main():
    lock = (ROOT / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == PROVENANCE['evaluator_revision'], revision
    subprocess.run(['git', '-C', str(REPO), 'diff', '--exit-code'], check=True)
    assert (ROOT / 'behavior_install_complete').exists(), 'Simulator installation incomplete'
    response = urllib.request.urlopen('http://127.0.0.1:8010/healthz', timeout=5)
    assert response.status == 200
    metadata = REPO / 'datasets/2026-challenge-task-instances/metadata/B100_task_misc.csv'
    with metadata.open(newline='') as file:
        mapping = {row['Task']: int(row['Task ID']) for row in csv.DictReader(file)}
    for task_id, task in TASKS:
        assert mapping[task] == task_id, (task, mapping[task], task_id)
    completed = []
    for task_id, task in TASKS:
        for index in range(10):
            output = ROOT / 'results' / task / f'public_{index:02d}'
            provenance = {**PROVENANCE, 'task': task, 'task_id': task_id, 'public_index': index}
            manifest = output / 'provenance.json'
            if output.exists():
                assert manifest.exists() and json.loads(manifest.read_text()) == provenance, output
                row = validate(output, task, index)
                completed.append(row)
                print(f'Reusing completed {task} public index {index}', flush=True)
                summarize(completed)
                continue
            output.mkdir(parents=True)
            manifest.write_text(json.dumps(provenance, indent=2) + '\n')
            command = [PYTHON, '-m', 'omnigibson.eval.eval', '--task-name', task,
                       '--host', '127.0.0.1', '--port', '8010', '--mode', 'public_test',
                       '--instance-indices', str(index), '--num-envs', '1', '--num-rollouts', '1',
                       '--env-wrapper', PROVENANCE['wrapper'], '--output-dir', str(output),
                       '--write-video', '--headless']
            (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            validation = task_id == 7 and index == 0
            write_status(status='validation_rollout' if validation else 'evaluating', task=task,
                         public_index=index, completed=len(completed), artifacts=str(output))
            print(f'{now()} Starting {task} public index {index}; artifacts {output}', flush=True)
            with (output / 'evaluator.log').open('x') as log:
                process = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                write_status(status='blocked', task=task, public_index=index, completed=len(completed),
                             error=f'Evaluator exited {process.returncode}', artifacts=str(output))
                raise RuntimeError(f'Evaluator failed; inspect {output}/evaluator.log')
            row = validate(output, task, index)
            completed.append(row)
            summarize(completed)
            if validation:
                (ROOT / 'validation_passed.json').write_text(json.dumps(row, indent=2) + '\n')
                print(f'End-to-end validation passed: {json.dumps(row)}', flush=True)
        task_summary = [entry for entry in summarize(completed) if entry['task'] == task][0]
        print(f'TASK FINISHED: {json.dumps(task_summary)}', flush=True)
    write_status(status='complete', completed=50, summary=str(ROOT / 'summary.json'))
    print('GROUP B COMPLETE: ' + str(ROOT / 'summary.json'), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        previous_path = ROOT / 'progress.json'
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        previous.pop('updated_at', None)
        previous.update(status='blocked', error=str(error))
        write_status(**previous)
        print(f'STOPPED: {error}', file=sys.stderr, flush=True)
        raise
