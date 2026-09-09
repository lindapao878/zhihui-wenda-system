db = db.getSiblingDB("kb001");
db.createUser({
  user: "zhihui_user",
  pwd: "REPLACE_WITH_PASSWORD",
  roles: [{ role: "readWrite", db: "kb001" }],
  authenticationRestrictions: []
});
