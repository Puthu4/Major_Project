import { useEffect, useState } from "react";

export default function BlockedUsers() {
  const [blockedUsers, setBlockedUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchBlocked = async () => {
    try {
      setLoading(true);
      const res = await fetch("http://localhost:5000/api/blocked-users");
      const data = await res.json();
      setBlockedUsers(data);
    } catch (err) {
      console.error("Error fetching blocked users:", err);
      alert("Failed to fetch blocked users.");
    } finally {
      setLoading(false);
    }
  };

  const unblockUser = async (email) => {
    if (!window.confirm(`Are you sure you want to unblock ${email}?`)) return;

    try {
      const res = await fetch("http://localhost:5000/api/unblock-user", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });

      const data = await res.json();

      if (!res.ok) {
        alert(data.message || "Failed to unblock user.");
        return;
      }

      alert(`✅ ${email} has been unblocked.`);
      fetchBlocked(); // refresh list
    } catch (err) {
      console.error("Unblock error:", err);
      alert("Failed to unblock user.");
    }
  };

  useEffect(() => {
    fetchBlocked();
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h1>🚫 Blocked Users</h1>

      {loading ? (
        <p>Loading blocked users...</p>
      ) : blockedUsers.length === 0 ? (
        <p>No blocked users 🎉</p>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "#f2f2f2" }}>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>Email</th>
              <th style={thStyle}>Reason</th>
              <th style={thStyle}>Blocked At</th>
              <th style={thStyle}>Action</th>
            </tr>
          </thead>

          <tbody>
            {blockedUsers.map((u, index) => (
              <tr key={index}>
                <td style={tdStyle}>{u.name || "Unknown"}</td>
                <td style={tdStyle}>{u.email}</td>
                <td style={tdStyle}>{u.blockedReason || "Not provided"}</td>
                <td style={tdStyle}>
                  {u.blockedAt
                    ? new Date(u.blockedAt).toLocaleString()
                    : "N/A"}
                </td>
                <td style={tdStyle}>
                  <button
                    onClick={() => unblockUser(u.email)}
                    style={unblockButtonStyle}
                  >
                    Unblock
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ======================= STYLES =======================

const thStyle = {
  padding: "10px",
  border: "1px solid #ddd",
  fontWeight: "bold",
};

const tdStyle = {
  padding: "10px",
  border: "1px solid #ddd",
};

const unblockButtonStyle = {
  padding: "6px 12px",
  background: "green",
  color: "white",
  border: "none",
  borderRadius: "5px",
  cursor: "pointer",
  fontSize: "14px",
};
