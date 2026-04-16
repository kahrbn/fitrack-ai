import datetime
from services.memory import save_json

def render_fitness_log(st, LOG_FILE):
    st.subheader("Log Workout")

    exercise = st.text_input("Exercise")
    sets = st.text_input("Sets/Reps")

    if st.button("Add"):
        entry = {
            "date": str(datetime.date.today()),
            "exercise": exercise,
            "sets": sets,
        }
        st.session_state.fitness_log.append(entry)
        save_json(LOG_FILE, st.session_state.fitness_log)
        st.success("Added!")

    for e in st.session_state.fitness_log:
        st.write(e)