# gjcr
Can models reason in code?

The main purpose of this experiment is to find out **how much the model is able to think in code?** This will be done by the concept of necessity and sufficiency, primarily to find the true “effect of code” to determine the output of the model.


Our team presents a research paper on test time sampling and selection procedures from a single large open-source model (GPT-oss_120b). The research compares three strategies: repeated sampling with basic aggregation, diversity-selective sampling with cosines, and search-like processes like beam search and tree-of-thought. Chain-of-thought (natural-language) results and code generation are treated differently: for code we take Python blocks and just do selection over those code snippets, while for language outputs we do comparisons over full responses. Selected candidates are combined with a simple scoring rule combining model confidence and response length, with the effect of confidence attenuated as the number of samples gets larger.