class Example(object):
    """A single training/test example for the roc dataset."""
    def __init__(self,
                 input_id,
                 obs1,obs2,hyps,labels,
                 ans = None,
                 adjacancy = None

                 ):
        self.input_id = input_id
        self.hyps = hyps
        self.ans = ans - 1
        self.adjacancy = adjacancy
        self.obs1 = obs1
        self.obs2 = obs2
        self.hyp2idx = dict([(hyp, i) for i, hyp in enumerate(hyps)])
        self.labels = labels

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 answer

    ):
        self.example_id = example_id
        try:
            self.choices_features = [
                {
                    'tokens': tokens,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'sentence_ind': sentence_ind,
                    'graph': graph,
                    'sentence_ids':graph_embedding
                }
                for tokens, input_ids, input_mask, sentence_ind, graph, graph_embedding in choices_features
            ]
        except: 
            self.choices_features = [
                {
                    'tokens': tokens,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'sentence_ind': sentence_ind,
                    'graph': graph
                }
                for tokens, input_ids, input_mask, sentence_ind, graph in choices_features
            ]   
        self.answer = answer
