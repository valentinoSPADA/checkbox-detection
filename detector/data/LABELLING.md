# Labelling by hand

The model-made labels in `annotations.npz` are ~90% accurate, and the residual errors are not
random: they are letter counters (`O`, `0`, `p`) confidently called empty boxes. That is a
disagreement about *what a checkbox is*, and more model labels cannot fix it — averaging more
of the same opinion does not move a concept. A few hundred human labels can.

## 1. Build the page

```bash
docker compose run --rm detector python -m training.make_labeling_task
```

Writes `detector/data/label.html` — one self-contained file, no server and no network. It
holds every candidate the detector surfaces (the same 165 / 101 / 99 / 115 the API returns per
page) plus ~35 per page that the classifier **rejected**, because a miss cannot be recorded by
any process that only looks at what the detector returned.

## 2. Label

Open it in a browser. Each candidate is shown twice: close, to see whether the interior carries
a mark, and wide, to see whether the thing is a checkbox or a table cell. **Only what is inside
the red rectangle is being judged** — a mark in a neighbouring box does not count.

    1  not a checkbox        2  empty box        3  marked box
    S  skip                  Z  undo

The model's own verdict is deliberately **not shown**. It is recorded in the export for
analysis, but a person shown a model's answer before giving their own is not an independent
labeller, and independence is the entire point of this pass.

Progress is written to the browser's local storage after every answer, so the tab can be closed
and the work continued later. Press **Descargar labels.json** at the end, or at any point.

Skipping is a real answer. A crop nobody is sure about should be absent from the training set,
not guessed at.

## 3. Import and retrain

```bash
python -m training.import_labels --labels labels.json --out data/merged.npz
python -m training.train --annotations data/merged.npz
```

The import prints where the hand labels **overruled** the model, split by what the model had
said and how confidently. That report is the actual finding — a pass that agrees everywhere
taught the model nothing and should be visible as such rather than counted as new data.

Model labels describing the same box as a hand label are dropped (overlap ≥ 0.35, not exact
coordinates — a box can shift a pixel between runs, and an exact rule would leave the model's
mistake beside the human's correction of it). Hand labels are repeated 4× relative to model
ones, because a few hundred crops at equal weight among tens of thousands contribute nothing
to the gradient.
