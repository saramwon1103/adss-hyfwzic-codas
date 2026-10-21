# Hướng dẫn cập nhật GitHub từ Linux

Repo đích: https://github.com/saramwon1103/adss-hyfwzic-codas.git.

ZIP khung được cung cấp có mọi file nội dung rỗng. Gói cập nhật điền implementation vào đúng cấu trúc đó và thêm một số module cần để chia rõ preprocess/model/simulator/CLI. Chưa truy cập/xác thực được current remote từ phiên tạo gói; nội dung mới phải được đối chiếu với clone thật, nhất là nếu Quyên đã commit code mới.

## 1. Giải nén payload ở ngoài Git clone

Giả sử tải file vào Downloads:

```bash
mkdir -p ~/Downloads/adss-update
unzip ~/Downloads/adss-hyfwzic-codas-updated.zip -d ~/Downloads/adss-update
```

Payload source nằm tại:

```text
~/Downloads/adss-update/adss-hyfwzic-codas
```

Đây là source để ghép, chưa phải clone đã xác thực của repo. Không chép thư mục bao ngoài vào trong clone gây lồng `repo/adss-hyfwzic-codas/src`.

## 2. Mở hoặc clone repo thật

Nếu chưa clone:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/saramwon1103/adss-hyfwzic-codas.git
cd adss-hyfwzic-codas
```

Nếu đã có clone, chỉ `cd` tới thư mục gốc clone của bạn. Không chạy lại git init, không đổi remote bừa và không force push.

Kiểm tra:

```bash
git remote -v
git status --short
git branch --show-current
```

Nếu có thay đổi chưa commit, commit chúng vào nhánh riêng hoặc stash cả untracked files trước khi cập nhật:

```bash
git stash push -u -m "Local work before DSS real-data update"
```

Lưu ý `stash -u` không cất file ignored. Helper chỉ cập nhật các file thuộc payload, không xóa file khác; các file bị thay đều có backup ở ngoài clone. Nếu file nguồn dữ liệu đã khác bản upload thì xem preview/diff để chọn phần giữ lại.

Lấy cập nhật remote rồi làm trên nhánh mới:

```bash
git pull --ff-only
git switch -c feature/mode2-real-data
```

Nếu nhánh này đã có, dùng một tên mới hoặc chuyển vào nhánh hiện có bằng `git switch feature/mode2-real-data`, kiểm tra status trước khi tiếp tục. Nếu pull báo diverged, dừng và xử lý nhánh hiện tại; không dùng --force để bỏ qua.

## 3. Preview và áp dụng payload

Helper chỉ dùng Python standard library và Git, không cần cài numpy/Streamlit trước.

Giả sử clone ở `~/projects/adss-hyfwzic-codas`:

```bash
python3 ~/Downloads/adss-update/adss-hyfwzic-codas/tools/apply_update.py \
  --target ~/projects/adss-hyfwzic-codas
```

Lệnh trên **chỉ liệt kê** ADD/UPDATE. Kiểm tra danh sách, đặc biệt các module nhóm đã có implementation riêng.

Áp dụng:

```bash
python3 ~/Downloads/adss-update/adss-hyfwzic-codas/tools/apply_update.py \
  --target ~/projects/adss-hyfwzic-codas --apply
```

Helper yêu cầu clone sạch, backup các file sắp bị ghi đè vào thư mục `adss-hyfwzic-codas-before-update-*` cạnh clone. Nó không chép Git history, venv, secrets hoặc generated outputs; không xóa file ngoài payload; giữ các dòng .gitignore cũ và bổ sung patterns còn thiếu. Helper không add/commit/push.

Sau khi apply:

```bash
cd ~/projects/adss-hyfwzic-codas
git diff --stat
git diff -- README.md app.py src
```

Nếu có thay đổi của Quyên cần giữ, dùng backup/diff để ghép lại trước khi commit. Thay cấu trúc không đồng nghĩa được bỏ qua code nhóm đã có.

## 4. Cài môi trường và kiểm thử

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest
python -m src.paper.reproduce_paper
python -m src.irrigation.run_irrigation --scenario all
python -m streamlit run app.py
```

Kết quả mong đợi: 55 passed; paper 13/13 và104/104; Mode2 source=real, 43 ngày. `Ctrl+C` để dừng UI trước khi commit. Nếu bạn đã có venv phù hợp thì kích hoạt lại, không cần tạo lại.

## 5. Commit và push nhánh lên GitHub

```bash
git add README.md .gitignore requirements.txt pyproject.toml app.py \
  src data docs tests tools \
  reproduce_paper.py run_irrigation.py prepare_real_data.py make_figures.py

git diff --cached --stat
git diff --cached --check
git commit -m "Use uploaded irrigation data in Mode 2 with src layout"
git push -u origin feature/mode2-real-data
```

GitHub sẽ có nhánh mới sau khi push thành công. Mở repo trên GitHub, chọn Compare & pull request và tạo PR vào nhánh chính của nhóm. Đây là cập nhật nhánh; code trên main chỉ đổi khi PR được merge.

Nếu repository bảo vệ main, tiếp tục qua PR thay vì push thẳng. Nếu nhóm quyết định merge sau khi review, dùng UI GitHub; hướng dẫn này không tự merge.

Generated outputs bị ignore, giúp tránh commit lại hàng trăm CSV mỗi lần chạy. Hai source CSV thực cần cho tái lập nằm trong data/irrigation/raw và được stage bình thường. Mỗi file khoảng 5 MB, không cần Git LFS cho gói chọn lọc này.

## 6. Khi Git yêu cầu xác thực

Nếu đã cài GitHub CLI:

```bash
gh auth login
gh auth setup-git
```

Chọn GitHub.com, HTTPS và đăng nhập bằng browser khi CLI hướng dẫn. Sau đó chạy lại clone/push. Không dán token vào chat hoặc nhúng token trong remote URL.

Nếu Git báo Author identity unknown, cấu hình riêng repo bằng tên/email GitHub của bạn:

```bash
git config user.name "TEN_GITHUB_CUA_BAN"
git config user.email "EMAIL_GITHUB_CUA_BAN"
```

Nếu `gh` chưa có, cài GitHub CLI theo tài liệu chính thức https://cli.github.com rồi đăng nhập. Hướng dẫn chính thức push: https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository.

## 7. Khi muốn hoàn tác trước khi commit

Clone sạch trước apply nghĩa là thay đổi tracked files trong `git diff` là của lượt apply. Có thể dùng `git restore` **cho từng đường dẫn đã kiểm tra** hoặc khôi phục nội dung từ backup. Không dùng `git clean -fd` hoặc reset --hard để xử lý một cách chung chung vì có thể mất phần mới của bạn.

Sau khi công việc trên nhánh ổn định, chỉ áp dụng lại stash nếu thật sự đã tạo stash ở bước 2. `git stash pop` có thể tạo conflict; ghép từng file và chạy lại tests.
