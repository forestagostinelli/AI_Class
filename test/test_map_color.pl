color(red).
color(blue).
color(green).

solution(WA, NT, SA, Q, NSW, V, T) :-
    color(WA), color(NT), \+ WA=NT, color(SA), \+ WA=SA, \+ NT=SA, color(Q), \+ NT=Q, \+ SA=Q,
    color(NSW), \+ Q=NSW, \+ SA=NSW, color(V), \+ SA=V, \+ NSW=V, color(T).

print_colors :- solution(WA, NT, SA, Q, NSW, V, T),
    maplist(write, ['WA: ', WA, ', NT: ', NT, ', SA: ', SA, ', Q: ', Q, ', NSW: ', NSW, ', V: ', V, ', T: ', T]).