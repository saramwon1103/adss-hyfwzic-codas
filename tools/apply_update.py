"""Preview/apply this tested payload to an existing clean Git clone; never push."""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT_FILES={'.gitignore','README.md','requirements.txt','pyproject.toml','app.py',
            'reproduce_paper.py','run_irrigation.py','prepare_real_data.py','make_figures.py'}
ROOT_DIRS={'src','data','docs','tests','tools'}
EXCLUDE_PARTS={'__pycache__','.pytest_cache','.git','.venv'}


def git_root(target):
    target=Path(target).resolve()
    p=subprocess.run(['git','-C',str(target),'rev-parse','--show-toplevel'],text=True,capture_output=True)
    if p.returncode or Path(p.stdout.strip()).resolve()!=target:
        raise ValueError('--target must be the root of an existing Git clone, not a nested directory.')
    return target


def merge_ignore(existing,incoming):
    if not existing:return incoming
    current=existing.decode('utf-8')
    known=set(current.splitlines())
    lines=[line for line in incoming.decode('utf-8').splitlines() if line and not line.startswith('#') and line not in known]
    return (current.rstrip()+'\n\n# Added by DSS real-data update\n'+'\n'.join(lines)+'\n').encode() if lines else existing


def plan_update(source,target):
    source=Path(source).resolve();target=git_root(target)
    if source==target:raise ValueError('Extract the update package outside the target Git clone.')
    plan=[]
    for file in sorted(source.rglob('*')):
        if not file.is_file() or file.is_symlink():continue
        rel=file.relative_to(source)
        if any(p in EXCLUDE_PARTS for p in rel.parts):continue
        if not ((len(rel.parts)==1 and rel.name in ROOT_FILES) or rel.parts[0] in ROOT_DIRS):continue
        dest=target/rel
        if not dest.resolve().is_relative_to(target) or dest.is_symlink():
            raise ValueError(f'Refusing a symlink outside target: {rel}')
        if dest.exists() and not dest.is_file():raise ValueError(f'File/directory conflict: {rel}')
        before=dest.read_bytes() if dest.exists() else None
        content=file.read_bytes()
        if rel.as_posix()=='.gitignore':content=merge_ignore(before or b'',content)
        if content!=before:plan.append((rel,content,before))
    return plan


def apply_plan(target,plan):
    target=git_root(target)
    status=subprocess.run(['git','-C',str(target),'status','--porcelain'],capture_output=True,text=True,check=True)
    if status.stdout.strip():
        raise ValueError('Target has uncommitted changes. Commit or stash them, then rerun; no files were changed.')
    if not plan:return None
    backup=Path(tempfile.mkdtemp(prefix=target.name+'-before-update-',dir=target.parent))
    # Back up every overwritten file before starting any payload writes.
    for rel,content,before in plan:
        if before is not None:
            saved=backup/rel;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(before)
    for rel,content,before in plan:
        dest=target/rel
        if dest.is_symlink() or not dest.resolve().is_relative_to(target):
            raise ValueError(f'Target changed to an unsafe path: {rel}')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    return backup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--apply',action='store_true',help='Without this flag, only show planned changes.')
    args=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    try:
        plan=plan_update(source,args.target)
        print(f'Planned updates: {len(plan)} files')
        for rel,content,before in plan:print(('ADD   ' if before is None else 'UPDATE')+' '+rel.as_posix())
        if args.apply:
            backup=apply_plan(args.target,plan)
            print('Applied. Backup:',backup if backup else '(no changes)')
            print('Review git diff, run tests, then commit/push from your clone.')
        else:print('Preview only. No target files changed. Rerun with --apply after review.')
    except (ValueError,subprocess.CalledProcessError) as error:
        p.exit(1,str(error)+'\n')

if __name__=='__main__':main()
