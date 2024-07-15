# MonoPara

Code for [Monotonic paraphrasing improves generalization of language model prompting](https://arxiv.org/pdf/2403.16038).

- To generate paraphrase with MonoPara, use the function [`run_PPL_greedy`](https://github.com/luka-group/MonoPara/blob/1173c635bda28344159638d3d4b151e015c7d84f/main_down_stream.py#L224) in `main_down_stream.py`.
- The logits ensemble is at [here](https://github.com/luka-group/MonoPara/blob/1173c635bda28344159638d3d4b151e015c7d84f/main_down_stream.py#L168). We manipulate the predicted logits at every decoding step.
- The larger `alpha` means more dependent on `logit_for_next_step_target`, which means assigning higher weight to the token that has the highest probability predicted by the target model, which is equivalent to lower perplexity wrt. target model. We set `alpha=0.5` for all the experiments.
- Ignore the `option` parameter and leave it as default.
- We use model `mistralai/Mistral-7B-Instruct-v0.1` as both target model and paraphrase model.
