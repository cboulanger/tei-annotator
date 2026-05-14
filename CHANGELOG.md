# [1.3.0](https://github.com/cboulanger/tei-annotator/compare/v1.2.0...v1.3.0) (2026-05-14)


### Features

* add INFO-level pipeline diagnostics for issue [#2](https://github.com/cboulanger/tei-annotator/issues/2) debugging ([81dc61e](https://github.com/cboulanger/tei-annotator/commit/81dc61ec435c7bfcb1b0ea036de8b1a475e7ada3))

# [1.2.0](https://github.com/cboulanger/tei-annotator/compare/v1.1.1...v1.2.0) (2026-05-14)


### Features

* add warning for span resolver context mismatches ([11fc401](https://github.com/cboulanger/tei-annotator/commit/11fc401dd0ca172306b94f7a0743d22d5a63f5a3)), closes [#2](https://github.com/cboulanger/tei-annotator/issues/2)

## [1.1.1](https://github.com/cboulanger/tei-annotator/compare/v1.1.0...v1.1.1) (2026-05-14)


### Bug Fixes

* merge overlapping spans from chunks to prevent text reordering ([4b612a1](https://github.com/cboulanger/tei-annotator/commit/4b612a1fb5c7fe0114580b453cb0f5c8e7d52ab2)), closes [#2](https://github.com/cboulanger/tei-annotator/issues/2)

# [1.1.0](https://github.com/cboulanger/tei-annotator/compare/v1.0.0...v1.1.0) (2026-05-14)


### Features

* **webservice:** fix timeout handling, reduce default LLM timeout to 60s ([d499dab](https://github.com/cboulanger/tei-annotator/commit/d499dabfbc04e3cb94aa34512b5b8d782e69c82b))

# 1.0.0 (2026-05-14)


### Bug Fixes

* add [@spaces](https://github.com/spaces).GPU decorator to satisfy ZeroGPU spaces check; graceful fallback when spaces not installed ([40d8c92](https://github.com/cboulanger/tei-annotator/commit/40d8c92be6089d52b468f9004582e9f21e7759b7))
* Add back gemini 2.0 flash model ([e7ad4b5](https://github.com/cboulanger/tei-annotator/commit/e7ad4b5e5334628e95851c0c9a02d53af04a0b44))
* Add rate limiter to Kisski connector ([cfaba49](https://github.com/cboulanger/tei-annotator/commit/cfaba49b966649d17401ea5af06c995f7ceda375))
* catch exceptions in do_evaluate to show error in UI instead of crashing ZeroGPU runtime ([82b98f7](https://github.com/cboulanger/tei-annotator/commit/82b98f75f16cfcc739133624e9074c34ca025d94))
* disable SSR mode in Gradio launch to prevent Node.js server crash on HF Spaces ([01465bb](https://github.com/cboulanger/tei-annotator/commit/01465bbc41a7cb7ebca30c566fd1cf659a933ef9))
* escape bare & in text nodes without double-encoding existing entities ([7a987f8](https://github.com/cboulanger/tei-annotator/commit/7a987f8743a595aadaeb3050c23fcc606053e834))
* explicitly set hardware: cpu-basic in Space metadata to suppress spaces.GPU check ([3c798ff](https://github.com/cboulanger/tei-annotator/commit/3c798ff051e7d404c85efa029213cde3a4f8a342))
* Fix config files ([0b7260e](https://github.com/cboulanger/tei-annotator/commit/0b7260eaa7ac5a61a50ce5c51e0f3f2a316565e1))
* increase [@spaces](https://github.com/spaces).GPU timeout to 300s to avoid GPU task abort on slow LLM calls ([a395ade](https://github.com/cboulanger/tei-annotator/commit/a395adecb93fee97c78579fc15b2129f7bd77a1d))
* Increase timeout ([4006797](https://github.com/cboulanger/tei-annotator/commit/4006797de91175d610ac26ea5f8e0cacbd317275))
* prompt rule improvements from 2026-05-08 evaluation experiments ([d26a27c](https://github.com/cboulanger/tei-annotator/commit/d26a27c9c5b9d92e2855636c1d2ceddd0d5aea82))
* remove local package install from requirements.txt (HF Spaces copies source directly) ([abc28a1](https://github.com/cboulanger/tei-annotator/commit/abc28a1feb52de3989fb1a6e0b83dd54081f32c9))
* rename EvaluateRequest.schema → schema_id, guard response construction, raise keepalive ([1c4c16a](https://github.com/cboulanger/tei-annotator/commit/1c4c16aeec1afc37180ed6cb8b3ad5efbe7e3912))
* replace editable install (-e .) with plain . in requirements.txt for HF Spaces compatibility ([0dcfc4c](https://github.com/cboulanger/tei-annotator/commit/0dcfc4c840ff72ea99f24c7d5188e48dee6e9359))
* sync batch size with sample size in gradio app ([1659d98](https://github.com/cboulanger/tei-annotator/commit/1659d98a3820a251581bf3c7217aff44c6b7abf2))


### Features

* Add batch size configuration in api and frontends ([b530e33](https://github.com/cboulanger/tei-annotator/commit/b530e336135d488aa77fb37d24da07346f993941))
* Add Gradio app for HF Spaces deployment ([331a802](https://github.com/cboulanger/tei-annotator/commit/331a80280d3a409237da960383ea7532698a19ae))
* Add registry to support any kind of inference provider ([f65b650](https://github.com/cboulanger/tei-annotator/commit/f65b6501074293efba4c9597f860e0b49874b997))
* Add security against malicious clients ([a1785f7](https://github.com/cboulanger/tei-annotator/commit/a1785f73c2c4212a6fec304217f2cce801e9c629))
* Add webservice for demonstration ([c3f33b6](https://github.com/cboulanger/tei-annotator/commit/c3f33b679e71429229bde937ae38a36689c7da53))
* cert="low" uncertain-boundary evaluation mechanism ([6826850](https://github.com/cboulanger/tei-annotator/commit/68268504ee31c64530bcc8e0fed9fb0bf816e08c))
* collect_hard_examples.py — find challenging gold examples via mini-batch evaluation ([ccfad34](https://github.com/cboulanger/tei-annotator/commit/ccfad34450269f03d4c17993081a6606a24c1e02))
* separate evaluation corpora from tests; add schema/corpus selection to webservice ([796d53e](https://github.com/cboulanger/tei-annotator/commit/796d53ee0f9c8a71675f57403436d46eff02453e))
* Support more providers ([18a607c](https://github.com/cboulanger/tei-annotator/commit/18a607c518e3c18a56948169cf31b3df6395ca44))
* **webservice:** show-examples mode, model status indicators, hard LLM timeout ([90e4492](https://github.com/cboulanger/tei-annotator/commit/90e4492189940156bbeb84158663b1eefc3be2bb))


### Performance Improvements

* cache blbl schema at module load instead of rebuilding per request ([4ba8656](https://github.com/cboulanger/tei-annotator/commit/4ba865620dd719b9972bbed58668a3240cba1d33))
