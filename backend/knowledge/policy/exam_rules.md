# CAT online exam: proctoring rules

This file is the proctoring knowledge base. The GraphRAG index (proctor/knowledge.py) turns it into a knowledge graph. When the AI judge reviews a flagged clip, it receives the rules that govern that flag and must cite the rule it applied.

Examiners can edit this file. Keep the format: a `## R-xx Title` heading, `key: value` lines, then the rule text. Then call `POST /proctor/admin/knowledge/rebuild` or restart the server.

The keys are:
- `applies_to`: the detector flags the rule governs
- `targets`: the objects, people or behaviours it is about
- `severity`: `minor`, `major` or `critical`
- `allowed`: exceptions, separated by `;`

## R-01 Eyes on the exam screen
applies_to: OFFSCREEN_SUSTAINED, REPEATED_GLANCES
targets: looking away, left, right, up
severity: major
allowed: a single glance away shorter than 2 seconds; looking up briefly while thinking; blinking or rubbing the eyes
The candidate must keep their eyes on the exam screen for the whole exam. Looking away for 3 seconds or more, or glancing repeatedly toward the same direction, indicates reading notes, a phone, a second screen or a person outside the camera view. Repeated glances to the same side are more serious than one long look, because they show a consistent source of information.

## R-02 Looking down at the desk or lap
applies_to: OFFSCREEN_SUSTAINED, REPEATED_GLANCES
targets: down, looking down, notes, phone, lap
severity: major
allowed: looking at the keyboard or mouse for less than 2 seconds; drinking water
Looking down for several seconds, or repeatedly, is the most common way to read a phone or notes hidden on the lap or desk below the camera. The exam is answered with the mouse only, so there is no need to look down at the keyboard for long.

## R-03 No mobile phones or smart devices
applies_to: PROHIBITED_OBJECT
targets: phone, smartwatch, earphones
severity: critical
allowed: a phone lying face-down out of reach that the candidate never touches; a phone visible only in the background of the room and not held
Mobile phones, smartwatches and earphones must not be held, used or looked at during the exam. A phone held in the hand, raised to the camera, or on the desk within reach is a violation. Holding a phone during the exam is treated as critical because it gives instant access to answers and messaging.

## R-04 No books, notes or paper
applies_to: PROHIBITED_OBJECT
targets: book, notes, paper
severity: major
allowed: books on a shelf or furniture in the background that the candidate never touches; a blank sheet shown to the camera when an examiner asks
Books, notebooks, printed or handwritten notes are not allowed on the desk or in the hands. A book lying on the desk below the candidate's chin, or held in the hands, is a violation. Background shelves are not.

## R-05 No second screen or second computer
applies_to: PROHIBITED_OBJECT
targets: laptop, second screen, tablet, tv
severity: critical
allowed: a switched-off television in the background of the room
Only the exam computer may be used. A second laptop, tablet, monitor or a television showing content that the candidate looks at is a violation.

## R-06 The candidate must be alone
applies_to: MULTIPLE_FACES, EXTRA_PERSON
targets: another person, second face
severity: critical
allowed: a person walking past in the background for less than 2 seconds without interacting; a face on a poster or photo in the room
No other person may be present in the room during the exam. A second face or body in the camera view, especially close to the candidate, suggests assistance or impersonation.

## R-07 No signals or help from others
applies_to: FOREIGN_HAND
targets: another person, hand, signalling
severity: critical
allowed: the candidate's own hand resting on the chin or face
A hand that does not belong to the candidate, such as one entering from the side or showing fingers or notes, indicates someone helping or signalling answers.

## R-08 Stay in the camera view
applies_to: NO_FACE
targets: leaving, face covered, absent
severity: major
allowed: face briefly hidden for under 3 seconds while adjusting the camera or drinking; poor lighting confirmed by the examiner
The candidate's face must stay visible to the camera. Leaving the seat, covering the face or turning fully away prevents proctoring and may hide a replacement or outside help.

## R-09 No talking during the exam
applies_to: VOICE_DETECTED
targets: candidate, talking, reading aloud, whispering
severity: major
allowed: coughing, sneezing or clearing the throat; a few words said to oneself without anyone answering; an approved accommodation (see R-13)
The candidate must not talk during the exam. Talking can mean the candidate is asking someone for answers, dictating questions to a helper, or taking part in a call. Reading questions aloud is also a violation, because it can leak exam content to someone listening.

## R-10 No other voices in the room
applies_to: VOICE_DETECTED
targets: another person, whispering, dictating answers
severity: critical
allowed: a voice clearly from another room, a television or the street that the candidate does not react to
Another person's voice near the candidate is heard while the candidate's own lips are still. This indicates someone in the room helping, dictating answers or reading from a source. It is the audio counterpart of R-06 and R-07.

## R-11 Background noise is not a violation
applies_to: VOICE_DETECTED, NO_FACE
targets: unattributed, background noise, television, traffic
severity: minor
allowed: all ordinary household or street noise
Noise the candidate does not create or react to is not malpractice. This includes traffic, a television in another room, family members in another room, or construction. When the speaker cannot be identified and the candidate keeps working normally, the flag should be marked suspicious for a human to check, not fraud.

## R-12 Repeated incidents escalate
applies_to: OFFSCREEN_SUSTAINED, REPEATED_GLANCES, PROHIBITED_OBJECT, VOICE_DETECTED, MULTIPLE_FACES, EXTRA_PERSON, FOREIGN_HAND, NO_FACE
targets: repeated, pattern, session history
severity: major
allowed: unrelated one-off flags spread across a long exam
Several confirmed incidents in the same session, or the same behaviour repeated after a warning popup, form a pattern. A pattern is stronger evidence of malpractice than any single incident and must be escalated to the examiner.

## R-13 Approved accommodations
applies_to: VOICE_DETECTED, OFFSCREEN_SUSTAINED
targets: accommodation, screen reader, assistive technology
severity: minor
allowed: speech or looking away that is required by an accommodation approved before the exam
Candidates with an approved accommodation, such as a screen reader, dictation software or a sign-language interpreter, are not in violation for behaviour the accommodation requires. The exam administrator records accommodations before the exam starts.

## R-14 AI verdicts are reviewed by a human
applies_to: OFFSCREEN_SUSTAINED, REPEATED_GLANCES, PROHIBITED_OBJECT, VOICE_DETECTED, MULTIPLE_FACES, EXTRA_PERSON, FOREIGN_HAND, NO_FACE
targets: examiner, human review, appeal
severity: minor
allowed: none
An AI verdict never ends an exam automatically. Every fraud or suspicious verdict is queued for an examiner, who confirms or dismisses it. Examiner decisions are recorded as past cases, so future AI judgements learn from them.
