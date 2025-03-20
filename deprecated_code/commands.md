## local

```bash
# to-be-created, think a tmux and prefix name for me
./ms_final.sh \
    -ns 10000 \
    -r 88888 \
    -p fast_overlap \
    -g 40 \
    -bo 0.35 \
    -ro 0.3

# iccp
./ms.sh -ns 10000 -r 88888



# resume
./ms_resume_allargs.sh \
    -ns 10000 \
    -r 88888 \
    -p myrun \
    -g 80 \
    -bo 0.25 \
    -ro 0.2
```


## 3090

```bash
# iccp
./ms.sh -ns 10000 -r 12345

# fast
./ms_resume_allargs.sh \
    -ns 10000 \
    -r 12345 \
    -p fast_without_overlap \
    -g 40 \
    -bo 0.25 \
    -ro 0.2

# overlap
./ms_resume_allargs.sh \
    -ns 10000 \
    -r 12345 \
    -p overlap \
    -g 80 \
    -bo 0.35 \
    -ro 0.3


# iccp100k
./ms_resume_allargs.sh \
    -ns 100000 \
    -r 12345 \
    -p myrun \
    -g 40 \
    -bo 0.35 \
    -ro 0.3

# iccp100k80
./ms_resume_allargs.sh \
    -ns 100000 \
    -r 12345 \
    -p more_basis_overlap \
    -g 80 \
    -bo 0.35 \
    -ro 0.3


```
