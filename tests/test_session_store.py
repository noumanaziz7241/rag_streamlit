from memory_agent.sessions.store import SessionStore


def test_create_and_list_sessions(temp_db_path: str):
    store = SessionStore(temp_db_path)
    session = store.create_session(user_id="user1", title="Demo")
    sessions = store.list_sessions("user1")
    assert len(sessions) == 1
    assert sessions[0].session_id == session.session_id
    assert sessions[0].title == "Demo"


def test_update_title_and_touch(temp_db_path: str):
    store = SessionStore(temp_db_path)
    session = store.create_session(user_id="user1")
    store.update_title(session.session_id, "Updated title")
    store.touch_session(session.session_id)
    updated = store.get_session(session.session_id)
    assert updated is not None
    assert updated.title == "Updated title"


def test_delete_session(temp_db_path: str):
    store = SessionStore(temp_db_path)
    session = store.create_session(user_id="user1")
    store.delete_session(session.session_id)
    assert store.get_session(session.session_id) is None
