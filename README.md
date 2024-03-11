### Usage ways:

#### 1. with Docker:

install:

```
docker build -t test .
```

run:

```
docker run --rm -it -v $(pwd):/app test poetry run python3 test/model.py
```

#### 2. with Poetry:

install:

```
poetry install
```

run:

```
poetry run python3 test/model.py
```

### Results:

#### Best mectric value ~ 0.65