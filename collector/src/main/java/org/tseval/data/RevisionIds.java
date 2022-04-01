package org.tseval.data;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializer;

import java.util.LinkedList;
import java.util.List;

public class RevisionIds {
    
    public String revision;
    public List<Integer> methodIds = new LinkedList<>();
    
    // Serialization
    public static JsonSerializer<RevisionIds> sSerializer = getSerializer();
    
    public static JsonSerializer<RevisionIds> getSerializer() {
        return (obj, type, jsonSerializationContext) -> {
            JsonObject jObj = new JsonObject();
            
            jObj.addProperty("revision", obj.revision);

            JsonArray aMethodIds = new JsonArray();
            for (int methodId : obj.methodIds) {
                aMethodIds.add(methodId);
            }
            jObj.add("method_ids", aMethodIds);
            
            return jObj;
        };
    }
}
