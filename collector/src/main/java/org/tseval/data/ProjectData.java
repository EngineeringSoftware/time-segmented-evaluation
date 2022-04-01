package org.tseval.data;

import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class ProjectData {

    public String name;
    public String url;
    public List<String> revisions = new LinkedList<>();
    public Map<String, List<String>> parentRevisions = new HashMap<>();
    public Map<String, String> yearRevisions = new HashMap<>();
    
    // Deserialization
    public static final JsonDeserializer<ProjectData> sDeserializer = getDeserializer();
    
    public static JsonDeserializer<ProjectData> getDeserializer() {
        return (json, type, context) -> {
            try {
                ProjectData obj = new ProjectData();
    
                JsonObject jObj = json.getAsJsonObject();
                obj.name = jObj.get("name").getAsString();
                obj.url = jObj.get("url").getAsString();
                
                // revisions
                for (JsonElement eRevision : jObj.get("revisions").getAsJsonArray()) {
                    obj.revisions.add(eRevision.getAsString());
                }

                // parent revisions
                JsonObject jObjParentRevisions = jObj.get("parent_revisions").getAsJsonObject();
                for (Map.Entry<String, JsonElement> entry : jObjParentRevisions.entrySet()) {
                    List<String> parentRevisions = new LinkedList<>();
                    for (JsonElement eParent : entry.getValue().getAsJsonArray()) {
                        parentRevisions.add(eParent.getAsString());
                    }
                    obj.parentRevisions.put(entry.getKey(), parentRevisions);
                }

                // year revisions
                JsonObject jObjYearRevisions = jObj.get("year_revisions").getAsJsonObject();
                for (Map.Entry<String, JsonElement> entry : jObjYearRevisions.entrySet()) {
                    obj.yearRevisions.put(entry.getKey(), entry.getValue().getAsString());
                }


                return obj;
            } catch (IllegalStateException e) {
                throw new JsonParseException(e);
            }
        };
    }
}
