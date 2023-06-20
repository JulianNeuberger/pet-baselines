from augment import base
from data import model
import itertools
from numpy.random import binomial, shuffle

# Shuffle within Segments - Satzebene


class Trafo90Step(base.AugmentationStep):

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()

        for sentence in doc.sentences:
            token_seq = []
            tag_seq = []
            pos_seq = []

            # generate token, tag and pos_tag list
            for token in sentence.tokens:
                token_seq.append(token.text)
                pos_seq.append(token.pos_tag)
                tag_seq.append(token.bio_tag)

            # compare whether both lists have the same length - if not error
            assert len(token_seq) == len(
                tag_seq
            ), "Lengths of token sequence and BIO-tag sequence should be the same"

            # we need the original indices of each tag - (indice, tag)
            # bsp: [(0, 'O'), (1, 'B-Actor'), (2, 'O'), (3, 'B-Activity'), (4, 'B-Activity Data'), (5, 'I-Activity Data'), (6, 'I-Activity Data'), (7, 'I-Activity Data'), (8, 'I-Activity Data'),...]
            tags = [(i, t) for i, t in enumerate(tag_seq)]

            # split tags into groups - [(indice, tag), (),...]
            # bsp: [[(0, 'O')], [(1, 'B-Actor')], [(2, 'O')], [(3, 'B-Activity')], [(4, 'B-Activity Data'), (5, 'I-Activity Data'), (6, 'I-Activity Data'), (7, 'I-Activity Data'), (8, 'I-Activity Data')],...]
            groups = [
                list(g)
                for k, g in itertools.groupby(tags, lambda s: s[1].split("-")[-1])
            ]

            # shuffle tokens in groups: dazu wird indices mit den indices der tokens je gruppe erstellt
            # und anschließend geshuffelt. Dann wird für jeden Token aus sentence.tokens der neue token.text und
            # token.pos_tag gesetzt.
            pos = 0  # position des in dem ursprünglichen Satz zu ersetzenden tokens - fortlaufend über alle groups
            for group in groups:
                indices = [i[0] for i in group]  # [0] ... [4,5,6,7,8] ...

                # shuffle indeice array
                if binomial(1, self.prob):
                    indices_shuffeld = shuffle(indices)
                else:
                    indices_shuffeld = indices

                # set shuffled tokens
                for i in range(len(group)):
                    sentence.tokens[pos].text = token_seq[indices_shuffeld[i]]
                    sentence.tokens[pos].pos_tag = pos_seq[indices_shuffeld[i]]
                    pos += 1
        return doc

