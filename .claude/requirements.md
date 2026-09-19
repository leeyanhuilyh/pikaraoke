## Features

### Pitch change

#### User story

- User needs to be able to change the pitch of the song
- User able to see current pitch relative to song's base pitch

#### Functional requirements
- Pitch change should be seamless (video continues playing, audio continues from where pitch change occurred)
- Pitch change should take no more than 5 seconds

#### Progress

##### live-pitch-shifting

##### pitch-resume-position

### Vocal track toggle

#### User story

- User should be able to toggle on and off the vocal track

#### Functional requirements

- Vocal track should be separate from the backing track
- Track artifact (both vocal and backing) should be kept to a minimum (at dev's discretion)
- On toggle, both vocal and backing tracks should continue seamlessly
- Track switch should take no more than 5 seconds

#### Progress

###### demucs-vocal-separation

##### vocal-reduction-filter

### Lyrics display

#### User story

- User should see lyrics for the song show up on screen at the right point in the track (or slightly before the words are sung)
- User should see both the current line and next line of the song
- Lyrics should show up on the left and right of the screen, similar to how it is in karaoke screens

#### Functional requirements
- Lyrics display should be synced up to when the singer actually starts singing
- If lyrics file doesn't exist in local database, it needs to be fetched and synced up to the track

#### Progress