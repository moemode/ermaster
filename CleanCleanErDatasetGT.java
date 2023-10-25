/*
* Copyright [2016-2020] [George Papadakis (gpapadis@yahoo.gr)]
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
 */
package org.scify.jedai.generalexamples;

import org.apache.log4j.BasicConfigurator;
import org.scify.jedai.datamodel.Attribute;
import org.scify.jedai.datamodel.EntityProfile;
import org.scify.jedai.datamodel.IdDuplicates;
import org.scify.jedai.datareader.entityreader.EntitySerializationReader;
import org.scify.jedai.datareader.entityreader.IEntityReader;
import org.scify.jedai.datareader.groundtruthreader.GtSerializationReader;
import org.scify.jedai.datareader.groundtruthreader.IGroundTruthReader;
import org.scify.jedai.utilities.datastructures.AbstractDuplicatePropagation;
import org.scify.jedai.utilities.datastructures.BilateralDuplicatePropagation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
/**
 *
 * @author G.A.P. II
 */
public class CleanCleanErDatasetGT {

    public static void main(String[] args) throws IOException {
        BasicConfigurator.configure();
        String mainFolder = "/home/v/Documents/4whpm32y47-7/Real Clean-Clean ER data/newDBPedia/dbpedia/";
        String[] groundTruthFiles = {mainFolder + "newDBPediaMatches"
        };
        String fileName = groundTruthFiles[0] + "out";
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        IGroundTruthReader gtReader = new GtSerializationReader(groundTruthFiles[0]);
        List<EntityProfile> l1 = new ArrayList<>();
        List<EntityProfile> l2 = new ArrayList<>();
        gtReader.getDuplicatePairs(l1, l2);
        final AbstractDuplicatePropagation duplicatePropagation = new BilateralDuplicatePropagation(gtReader.getDuplicatePairs(null));
        Set<IdDuplicates> duplicates = duplicatePropagation.getDuplicates();
        for(IdDuplicates dup :duplicates) {
            writer.write(dup.getEntityId1() + "," + dup.getEntityId2() + "\n");
        }
        System.out.println("Existing Duplicates\t:\t" + duplicatePropagation.getDuplicates().size());
        writer.close();

    }

}
