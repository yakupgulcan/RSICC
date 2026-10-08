#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.env = os.environ
        self.env['LC_ALL'] = 'en_US.UTF_8'
        self.meteor_cmd = ['java', '-Duser.language=en', '-Duser.country=US', '-jar', '-Xmx2G', METEOR_JAR, \
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                                         cwd=os.path.dirname(os.path.abspath(__file__)), \
                                         stdin=subprocess.PIPE, \
                                         stdout=subprocess.PIPE, \
                                         stderr=subprocess.PIPE,
                                         env=self.env, universal_newlines=True, bufsize=1)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):

        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        try:
            for i in range(len(res)):
                assert (len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            # Send to METEOR
            self.meteor_p.stdin.write(eval_line + '\n')
            self.meteor_p.stdin.flush()

            # Collect segment scores
            for i in range(len(res)):
                line = self.meteor_p.stdout.readline().strip()
                try:
                    score = float(line)
                except ValueError:
                    # If an unexpected stat line was in buffer, read next line
                    line = self.meteor_p.stdout.readline().strip()
                    try:
                        score = float(line)
                    except ValueError:
                        score = 0.0
                scores.append(score)

            # Final score
            line = self.meteor_p.stdout.readline().strip()
            try:
                final_score = float(line)
            except ValueError:
                final_score = float(sum(scores) / len(scores)) if scores else 0.0
        finally:
            self.lock.release()

        return final_score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('\n', ' ').replace('\r', ' ').strip()
        if not hypothesis_str:
            hypothesis_str = "empty"
        clean_refs = []
        for ref in reference_list:
            r = ref.replace('|||', '').replace('\n', ' ').replace('\r', ' ').strip()
            if r:
                clean_refs.append(r)
        if not clean_refs:
            clean_refs = ["empty"]
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(clean_refs), hypothesis_str))
        self.meteor_p.stdin.write(score_line + '\n')
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().strip()

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()