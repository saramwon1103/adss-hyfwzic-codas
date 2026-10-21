from pathlib import Path
import subprocess
import pytest
from tools.apply_update import plan_update,apply_plan


def tracked_clone(path):
    path.mkdir()
    subprocess.run(['git','init','-q',str(path)],check=True)
    (path/'app.py').write_text('old code\n')
    (path/'.gitignore').write_text('company-local.txt\n')
    subprocess.run(['git','-C',str(path),'add','.'],check=True)
    subprocess.run(['git','-C',str(path),'-c','user.name=Test','-c','user.email=test@localhost','commit','-qm','fixture'],check=True)
    return path


def test_update_preview_backup_and_keep_user_ignore(tmp_path):
    target=tracked_clone(tmp_path/'clone')
    payload=tmp_path/'payload';payload.mkdir()
    (payload/'app.py').write_text('new code\n')
    (payload/'.gitignore').write_text('.venv/\n.env\n')
    (payload/'outputs').mkdir();(payload/'outputs/result.csv').write_text('not copied\n')
    plan=plan_update(payload,target)
    assert (target/'app.py').read_text()=='old code\n'
    backup=apply_plan(target,plan)
    assert (target/'app.py').read_text()=='new code\n'
    assert (backup/'app.py').read_text()=='old code\n'
    assert 'company-local.txt' in (target/'.gitignore').read_text()
    assert '.venv/' in (target/'.gitignore').read_text()
    assert not (target/'outputs/result.csv').exists()


def test_dirty_clone_is_not_overwritten(tmp_path):
    target=tracked_clone(tmp_path/'clone')
    (target/'app.py').write_text('my uncommitted change\n')
    payload=tmp_path/'payload';payload.mkdir();(payload/'app.py').write_text('new\n')
    with pytest.raises(ValueError,match='uncommitted'):
        apply_plan(target,plan_update(payload,target))
    assert (target/'app.py').read_text()=='my uncommitted change\n'
