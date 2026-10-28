# Telltale

Upload a voice clip and get a speech model's read on whether it is human or synthetic.

## Run it (Windows PowerShell)

```
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000

The first start downloads the model (a few hundred MB) from Hugging Face, so it needs internet and a minute or two. After that it is cached. Watch the terminal: when it says "Model ready" you can analyse clips. `http://127.0.0.1:8000/api/health` tells you if the model loaded, and if not, why.

## What changed compared with the original

- **Any audio format works.** The page decodes MP3, M4A, WAV, OGG and FLAC in the browser, converts to 16 kHz mono WAV, then uploads. Format and sample-rate mismatches were a likely cause of errors on new files.
- **Shorter limits.** Minimum 2 s, maximum 15 s. A longer clip is not rejected: the first 15 s are analysed and the page says so. Change with `MIN_SECONDS` and `MAX_SECONDS`.
- **Long clips are split** into segments of at most 13 s (the model's training range) and averaged. If segments clearly disagree the result is "Can't tell".
- **Class labels are read from the model config**, so AI vs human scores cannot be swapped.
- **New look**, no "AI" branding anywhere. The product name is Telltale; rename it in `static/index.html`.

## Settings (environment variables, all optional)

`MIN_SECONDS`, `MAX_SECONDS`, `SEGMENT_SECONDS`, `INCONCLUSIVE_MARGIN`, `MODEL_ID`, `MODEL_REVISION`, `SCORE_TEMPERATURE`.

### Score temperature (important)

`SCORE_TEMPERATURE` defaults to `1.0`, which leaves the model's raw softmax scores unchanged. A value below `1.0` sharpens scores (moves them farther from 50%); a value above `1.0` softens them. This changes the displayed score and the verdict decision, but does **not** improve detection or make the output a calibrated probability. For example, a raw 60% score can become roughly 77% with a sufficiently low temperature, depending on the model logits. Do not present an arbitrary temperature-adjusted score as a real probability. For a defensible setting, choose temperature using labelled validation recordings, then report results on separate held-out data. The UI shows the raw score under Technical details so it is possible to compare both.

## Limits you should state in your report

The scores are softmax outputs, not calibrated probabilities. The model card reports about 95% accuracy on English audio, unverified, so accuracy on other languages, phone-quality audio or newer voice-cloning tools is unknown. Do not treat a result as proof of anything. To measure it properly, collect your own labelled real and fake clips (including your friend's non-English data), run them through, and report the confusion matrix.


## Verdict thresholds

The verdict is based on the mean adjusted AI score across audio segments:

- Above `AI_THRESHOLD` (default `0.55`): `synthetic`
- Below `HUMAN_THRESHOLD` (default `0.45`): `human`
- Between those thresholds, inclusive: `inconclusive`

Set these as environment variables if needed. These thresholds only change the verdict, not the model's raw score or detection accuracy. `SCORE_TEMPERATURE` changes score sharpness and should not be treated as calibration without labelled validation data.
