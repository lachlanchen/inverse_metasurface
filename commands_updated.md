# commands_updated.md

## Local

### 1) Overlap run (originally `fast_overlap`)
- **Final TMUX Session Name**: `iccpOv10kG40`
- **Current Command** (unchanged):
  ```bash
  ./ms_final.sh \
      -ns 10000 \
      -r 88888 \
      -p fastOv \
      -g 40 \
      -bo 0.35 \
      -ro 0.3
  ```
- **Suggested New Prefix**: `iccpOv10kG40`  
  *(Because ns=10000, bo=0.35, ro=0.3 => Overlap, G=40.)*

---

### 2) iccp (using `ms.sh`, not resumable)
- **Final TMUX Session Name**: `iccpNoOv10kG80`
- **Current Command** (unchanged):
  ```bash
  ./ms.sh -ns 10000 -r 88888
  ```
- **Suggested New Prefix**: *(none, `ms.sh` is not resumable.)*  
  If you ever add `-p`, a consistent name might be `iccpNoOv10kG80`.

---

### 3) resume (prefix=`myrun`, no overlap, G=80)
- **Final TMUX Session Name**: `iccpNoOv10kG80Resume`
- **Current Command** (unchanged):
  ```bash
  ./ms_resume_allargs.sh \
      -ns 10000 \
      -r 88888 \
      -p myrun \
      -g 80 \
      -bo 0.25 \
      -ro 0.2
  ```
- **Suggested New Prefix**: `iccpNoOv10kG80R`  
  *(ns=10000, no overlap => bo=0.25, ro=0.2, G=80, plus R for “resume”.)*

---

## 3090

Below are the **original** commands with no changes. Next to each, we note the **final** session name you’ve set and a **unified prefix** suggestion if you ever revise them.

### 1) iccp
- **Final TMUX Session Name**: `iccp10kG80NoOv`
- **Current Command**:
  ```bash
  ./ms.sh -ns 10000 -r 12345
  ```
- **Suggested Prefix**: *(none, `ms.sh` not resumable.)*  
  If it were, we might pick `iccpNoOv10kG80`.

---

### 2) fast
- **Final TMUX Session Name**: `iccp10kG40NoOv`
- **Current Command**:
  ```bash
  ./ms_resume_allargs.sh \
      -ns 10000 \
      -r 12345 \
      -p fast_without_overlap \
      -g 40 \
      -bo 0.25 \
      -ro 0.2
  ```
- **Suggested New Prefix**: `iccpNoOv10kG40`  
  *(No overlap, 10k, G=40.)*

---

### 3) overlap
- **Final TMUX Session Name**: `iccp10kG80Ov`
- **Current Command**:
  ```bash
  ./ms_resume_allargs.sh \
      -ns 10000 \
      -r 12345 \
      -p overlap \
      -g 80 \
      -bo 0.35 \
      -ro 0.3
  ```
- **Suggested New Prefix**: `iccpOv10kG80`

---

### 4) iccp100k
- **Final TMUX Session Name**: `iccp100kG40Ov`
- **Current Command**:
  ```bash
  ./ms_resume_allargs.sh \
      -ns 100000 \
      -r 12345 \
      -p myrun \
      -g 40 \
      -bo 0.35 \
      -ro 0.3
  ```
- **Suggested New Prefix**: `iccpOv100kG40`  
  *(Overlap, 100k, G=40.)*

---

### 5) iccp100k80
- **Final TMUX Session Name**: `iccp100kG80Ov`
- **Current Command**:
  ```bash
  ./ms_resume_allargs.sh \
      -ns 100000 \
      -r 12345 \
      -p more_basis_overlap \
      -g 80 \
      -bo 0.35 \
      -ro 0.3
  ```
- **Suggested New Prefix**: `iccpOv100kG80`

---
